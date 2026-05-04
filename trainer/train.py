import os
import csv
import random
import sys
import time
import numpy
import gc
import json
from PIL import Image
import traceback
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from trainer.lora import LoRANetwork, LycorisNetwork
from trainer import trainer, dataset
from trainer.anima_support import (
    AnimaFlowScheduler,
    AnimaTextModel,
    anima_forward,
    expand_cond,
    move_cond_to_device,
)
from trainer.fd_loss import FDLossManager
from pprint import pprint
from typing import Optional
from accelerate.utils import set_seed

# Module-level FD-Loss manager reference, set during training so the Gradio UI
# can access the interactive cluster panel while training is running.
_fd_manager: Optional[FDLossManager] = None

# Pause-and-inspect mechanism for interactive guidance.
# When _fd_pause_step is set, the training loop will pause at that step
# and wait for the user to inspect clusters and perform guidance actions.
_fd_pause_step: int = 0          # step at which to pause (0 = no pause)
_fd_last_pause_step: int = 0     # last step that was paused (preserved for "continuing from step" message)
_fd_pause_interval: int = 0      # pause every N steps (0 = disabled)
_fd_resume_signal: bool = False  # set True by UI to resume training
_fd_paused: bool = False         # True while training is paused
_fd_just_resumed: bool = False   # transient flag: True briefly after resume so get_pause_status_html shows "Resumed"

# Add sd-scripts root to path for library imports
_TRAINTRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SD_SCRIPTS_ROOT = os.environ.get("SD_SCRIPTS_PATH") or os.path.dirname(_TRAINTRAIN_DIR)
if _SD_SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SD_SCRIPTS_ROOT)

try:
    from modules import shared
    _HAS_WEBUI = True
except ImportError:
    _HAS_WEBUI = False

MAX_DENOISING_STEPS = 1000
ML = "LoRA"

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

stoptimer = 0

CUDA = torch.device("cuda:0")

queue_list = []
current_name = None


def get_name_index(wanted):
    for i, name in enumerate(trainer.all_configs):
        if name[0] == wanted:
            return i


def queue(*args):
    global queue_list
    name_index = get_name_index("save_lora_name") + 4
    dup = args[name_index] == current_name
    for queue in queue_list:
        if queue[name_index] == args[name_index]:
            dup = True
    if dup:
        return "Duplicated LoRA name! Could not add to queue."
    queue_list.append(args)
    return "Added to Queue"


def get_del_queue_list(del_name=None):
    global queue_list
    name_index = get_name_index("save_lora_name")
    out = []
    del_index = None
    for i, q in enumerate(queue_list):
        data = [*q[1:-2]]
        name = data[name_index + 3]
        data = [name] + data
        if del_name and name == del_name:
            del_index = i
        else:
            out.append(data)
    if del_index:
        del queue_list[del_index]
    return out


def setcurrentname(args):
    name_index = get_name_index("save_lora_name") + 4
    global current_name
    current_name = args[name_index]


def train(*args):
    """Legacy positional-args entry point (kept for backward compat)."""
    if not args[0]:
        setcurrentname(args)
    result = train_main(*args)
    while len(queue_list) > 0:
        settings = queue_list.pop(0)
        result += "\n" + train_main(*settings)
    return result


def train_named(jsononly: bool, mode: str, modelname: str, vaename: str,
                config: dict, images=None) -> str:
    """
    Named-dict entry point.  Accepts a structured config dict instead of
    a fragile positional-args list.  JSON files saved by the trainer will
    still have the same key names for cross-branch compatibility.

    Args:
        jsononly: If True, only save preset JSON.
        mode: "LoRA", "ADDifT", or "Multi-ADDifT".
        modelname: Path to the model.
        vaename: Path to the VAE.
        config: Named dict of configuration values.
        images: Optional [orig_image_data, targ_image_data].
    """
    global current_name
    current_name = config.get("save_lora_name", "untitled")
    result = _train_impl(jsononly, mode, modelname, vaename, config, images)
    return result


def train_main(jsononly, mode, modelname, vaename, *args):
    """Legacy positional-args entry point (kept for backward compat)."""
    global _fd_pause_step, _fd_pause_interval, _fd_resume_signal, _fd_paused
    # Convert positional args to named dict
    config = {}
    for i, sets in enumerate(trainer.all_configs):
        if i < len(args):
            config[sets[0]] = args[i]
    images = args[len(trainer.all_configs) + 1:] if len(args) > len(trainer.all_configs) + 1 else None
    t = trainer.Trainer(jsononly, modelname, vaename, mode, config, images)
    return _train_impl(jsononly, mode, modelname, vaename, config, images, t)


def _train_impl(jsononly, mode, modelname, vaename, config, images=None, t=None):
    """Shared training logic used by both train() and train_named()."""
    global _fd_pause_step, _fd_pause_interval, _fd_resume_signal, _fd_paused
    if t is None:
        t = trainer.Trainer(jsononly, modelname, vaename, mode, config, images)

    if jsononly:
        return "Preset saved"

    if t.isfile:
        return "File exist!"

    if modelname == "":
        return "No Model Selected."

    print(" Start Training!")

    # ------------------------------------------------------------------ #
    # Load Anima-specific libraries                                        #
    # ------------------------------------------------------------------ #
    from library import anima_utils, qwen_image_autoencoder_kl

    t.sd_typer()

    # ------------------------------------------------------------------ #
    # Load VAE                                                             #
    # ------------------------------------------------------------------ #
    print(f"Loading VAE from {vaename}")
    vae = qwen_image_autoencoder_kl.load_vae(vaename, device="cpu", disable_mmap=True)
    vae = vae.to(CUDA, dtype=t.train_model_precision)
    vae.requires_grad_(False)
    vae.eval()
    t.vae = vae

    # ------------------------------------------------------------------ #
    # Load Qwen3 text encoder + tokenizers                                 #
    # ------------------------------------------------------------------ #
    qwen3_path = getattr(t, "qwen3_path", "")
    print(f"Loading Qwen3 text encoder from {qwen3_path}")
    qwen3_encoder, _ = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=t.train_model_precision, device="cpu"
    )
    qwen3_encoder = qwen3_encoder.to(CUDA)
    qwen3_encoder.requires_grad_(False)
    qwen3_encoder.eval()

    t5_tokenizer_path = getattr(t, "t5_tokenizer_path", None) or None

    t.text_model = AnimaTextModel(
        qwen3_encoder,
        qwen3_path,
        t5_tokenizer_path,
        device=CUDA,
        dtype=t.train_model_precision,
    )

    # ------------------------------------------------------------------ #
    # Load Anima DiT                                                       #
    # ------------------------------------------------------------------ #
    attn_mode = "torch"
    print(f"Loading Anima DiT from {modelname}")
    dit = anima_utils.load_anima_model(
        device=CUDA,
        dit_path=modelname,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device=CUDA,
        dit_weight_dtype=t.train_model_precision,
    )
    dit.requires_grad_(False)
    dit.eval()
    t.unet = dit  # raw DiT — LoRA modules patch this directly

    if t.use_gradient_checkpointing:
        dit.train()
        # Anima DiT gradient checkpointing (if supported)
        if hasattr(dit, "enable_gradient_checkpointing"):
            dit.enable_gradient_checkpointing()
        t.text_model.train()
        t.text_model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------ #
    # Encode prompts                                                       #
    # ------------------------------------------------------------------ #
    trigger = getattr(t, 'lora_trigger_word', '') or ''
    if t.mode in ("ADDifT", "Multi-ADDifT"):
        # orig = source/before condition, targ = destination/after condition.
        # For ADDifT single-pair, diff_target_name is the "before" text prompt.
        # For Multi-ADDifT, per-pair conditioning comes from batch captions;
        # these serve as fallbacks.
        before_text = getattr(t, 'diff_target_name', '') or ''
        t.orig_cond, _ = text2cond(t, before_text if before_text else trigger)
        t.targ_cond, _ = text2cond(t, trigger)
    else:
        t.orig_cond, _ = text2cond(t, trigger)
        t.targ_cond = t.orig_cond
    t.un_cond, _ = text2cond(t, '')

    # ------------------------------------------------------------------ #
    # Noise scheduler (flow matching)                                      #
    # ------------------------------------------------------------------ #
    t.noise_scheduler = AnimaFlowScheduler()

    # ------------------------------------------------------------------ #
    # Accelerator                                                          #
    # ------------------------------------------------------------------ #
    t.a = trainer.make_accelerator(t)
    t.unet = t.a.prepare(t.unet)

    if 0 > t.train_seed:
        t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)

    # Store helpers on t for dataset
    t.text2cond = text2cond
    t.image2latent = image2latent

    try:
        if t.mode == ML:
            result = train_lora(t)
        elif t.mode == "ADDifT" or t.mode == "Multi-ADDifT":
            result = train_diff2(t)
        else:
            result = "Test mode"

        print("Done.")
    except Exception as e:
        print(traceback.format_exc())
        result = f"Error: {e}"

    del t
    flush()

    return result


# --------------------------------------------------------------------------- #
# Train modes                                                                  #
# --------------------------------------------------------------------------- #

def train_lora(t):
    global stoptimer
    global _fd_pause_interval, _fd_paused, _fd_resume_signal, _fd_pause_step, _fd_last_pause_step
    stoptimer = 0

    t.a.print("Preparing image latents and text-conditional...")
    dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = t.a.prepare(t.dataloader)

    t.a.print("Train Anima LoRA Start")

    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    loss_ema = None
    loss_velocity = None

    _train_hybrid = getattr(t, 'train_hybrid_mode', False)

    # --- FD-Loss setup -------------------------------------------------------
    _use_fd_loss = getattr(t, 'fd_loss_enable', False)
    fd_manager = None
    if _use_fd_loss:
        t.a.print("Initializing FD-Loss...")
        fd_repr_models = getattr(t, 'fd_repr_models', 'dinov2_vitb14').strip()
        fd_repr_models = [m.strip() for m in fd_repr_models.split(",") if m.strip()]
        fd_queue_size = getattr(t, 'fd_queue_size', 50000)
        fd_queue_mode = getattr(t, 'fd_queue_mode', 'online_accum')
        fd_ema_beta = getattr(t, 'fd_ema_beta', 0.9999)
        fd_weight = getattr(t, 'fd_loss_weight', 0.1)
        fd_fid_norm_eps = getattr(t, 'fd_fid_norm_eps', 1e-6)
        # --- New: intelligent queue management parameters ---
        fd_eviction_mode = getattr(t, 'fd_eviction_mode', 'fifo')
        fd_guidance_strength = getattr(t, 'fd_guidance_strength', 0.5)
        fd_n_clusters = getattr(t, 'fd_n_clusters', 20)
        fd_cluster_log_interval = getattr(t, 'fd_cluster_log_interval', 0)  # 0 = disabled
        fd_store_source_images = getattr(t, 'fd_store_source_images', False)
        fd_enqueue_generated = getattr(t, 'fd_enqueue_generated', True)
        fd_pause_interval = getattr(t, 'fd_pause_interval', 0)  # 0 = disabled
        # Warmup: skip enqueuing generated features for the first N training steps.
        # During warmup, FD-Loss is still computed against the pre-filled real data
        # reference, but the model's own (noisy) outputs are not injected into the
        # queue — preventing early garbage from polluting the reference distribution.
        # The effective warmup is fd_warmup_steps * batch_size * grad_accum steps
        # worth of generated features that would otherwise contaminate the queue.
        fd_warmup_steps = getattr(t, 'fd_warmup_steps', 0)  # 0 = disabled
        # ----------------------------------------------------
        # Set pause interval on module-level variable for UI access
        _fd_pause_interval = fd_pause_interval

        fd_manager = FDLossManager(
            repr_models=fd_repr_models,
            queue_size=fd_queue_size,
            queue_mode=fd_queue_mode,
            ema_beta=fd_ema_beta,
            weights=[fd_weight] * len(fd_repr_models),
            fid_norm_eps=fd_fid_norm_eps,
            device=CUDA,
            # --- New ---
            eviction_mode=fd_eviction_mode,
            guidance_strength=fd_guidance_strength,
            n_clusters=fd_n_clusters,
            store_source_images=fd_store_source_images,
            enqueue_generated=fd_enqueue_generated,
        )
        t.a.print(f"  FD-Loss judges: {fd_repr_models}, queue_size={fd_queue_size}, mode={fd_queue_mode}, eviction={fd_eviction_mode}")

        # Expose to Gradio UI via module-level reference
        global _fd_manager
        _fd_manager = fd_manager

        # Pre-fill queues with real data so the covariance estimate is
        # non-degenerate from step 1 (avoids eigendecomposition failures).
        # In texture mode, pass mask_key="mask" so only the crop region is
        # enqueued — the frequency-matched noise background is excluded.
        _pf_mask_key = "mask" if getattr(t, 'texture_mode', False) else None
        t.a.print("  Pre-filling FD-Loss queues with training data...")
        fd_manager.prefill_from_dataloader(t.dataloader, t.vae, mask_key=_pf_mask_key)
        t.a.print("  FD-Loss queues pre-filled.")

        # Log initial cluster structure if interval is set
        _fd_cluster_step = fd_cluster_log_interval
        if _fd_cluster_step > 0:
            try:
                _html = fd_manager.get_cluster_summary_html()
                t.a.print(f"  Initial queue clusters:\n{_html}")
            except Exception:
                pass
    # -------------------------------------------------------------------------

    # VAE must stay alive for JIT encoding: texture_mode OR hybrid mode OR fd_loss
    if not getattr(t, 'texture_mode', False) and not _train_hybrid and not _use_fd_loss:
        del t.vae
        if "BASE" not in t.network_blocks:
            del t.text_model

    flush()

    # Parse timestep curriculum schedule from inline config text (optional)
    _ts_schedule = _parse_ts_schedule_text(getattr(t, 'train_ts_schedule', '') or '')
    if _ts_schedule:
        print(f"Timestep schedule loaded: {len(_ts_schedule)} entries"
              + (", hybrid mode active" if _train_hybrid else ""))

    # Prime hybrid processing mode before the first batch is fetched
    def _set_hybrid_mode(step_pct):
        if not _train_hybrid:
            return
        mode = ""
        if _ts_schedule:
            mode = _resolve_ts_entry(_ts_schedule, step_pct)[3]
        t.hybrid_processing_mode = mode or "texture"

    _set_hybrid_mode(0.0)

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1 = batch["cond1"] if "cond1" in batch else None

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Resolve timestep range and optional LR override from schedule (or static config)
                step_pct = pbar.n / max(1, t.train_iterations - 1)
                if _ts_schedule is not None:
                    _entry = _resolve_ts_entry(_ts_schedule, step_pct)
                    ts_lo, ts_hi, _lr_override = _entry[1], _entry[2], _entry[4]
                    if _lr_override is not None:
                        for pg in optimizer.param_groups:
                            pg['lr'] = _lr_override
                else:
                    ts_lo, ts_hi = t.train_min_timesteps, t.train_max_timesteps
                ts_lo = max(0, ts_lo)
                ts_hi = max(ts_lo + 1, min(1000, ts_hi))

                dist_type  = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                dist_params = getattr(t, 'train_ts_dist_params', '') or ''
                n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
                timesteps = _sample_timesteps(ts_lo, ts_hi, n_ts, CUDA, dist_type, dist_params)
                timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

                noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)

                # Resolve conditioning
                if conds1 is None:
                    conds1 = expand_cond(t.orig_cond, batch_size)
                elif isinstance(conds1, str) or (isinstance(conds1, list) and isinstance(conds1[0], str)):
                    conds1, _ = t.text_model.encode_text(conds1 if isinstance(conds1, list) else [conds1])
                elif isinstance(conds1, (tuple, list)):
                    conds1 = move_cond_to_device(conds1, CUDA, t.train_lora_precision)

                with network, t.a.autocast():
                    model_pred = anima_forward(t, noisy_latents, timesteps, conds1)

                # Flow matching loss target: velocity = noise - latents
                velocity_target = (noise - latents).to(torch.float32)
                train_mask = batch.get("mask")
                if train_mask is not None:
                    train_mask = train_mask.to(CUDA)

                loss, loss_ema, loss_velocity = process_loss(
                    t, model_pred, velocity_target, timesteps, loss_ema, loss_velocity,
                    mask=train_mask,
                )

                # --- FD-Loss: perceptual quality via differentiable FID ----------
                if _use_fd_loss and fd_manager is not None:
                    # Reconstruct predicted clean latents from velocity prediction
                    # Flow matching: noisy = (1-t)*clean + t*noise
                    # velocity = noise - clean  (predicted)
                    # So: pred_clean = noisy_latents - timesteps_normalized * model_pred
                    # where timesteps_normalized = timesteps / 1000
                    ts_norm = timesteps.float() / 1000.0  # [B] in [0, 1]
                    # pred_velocity = model_pred (what the model predicts)
                    # clean = noisy - ts * velocity  (rearranged from noisy = clean + ts * velocity)
                    # But flow matching uses: noisy = (1-t)*clean + t*noise
                    # So: velocity = noise - clean
                    #     noisy = clean + ts * velocity
                    #     clean = noisy - ts * velocity
                    pred_clean = noisy_latents.float() - ts_norm.view(-1, 1, 1, 1) * model_pred.float()

                    # Decode predicted clean latents to pixels [0, 1]
                    pred_pixels = latent2pixels(t, pred_clean)

                    # In texture mode, the batch contains a full canvas with frequency-matched
                    # noise background + a placed crop.  We must feed ONLY the crop region to
                    # FD-Loss — the background noise would pollute the feature queue and make
                    # the FID comparison meaningless (comparing noise distributions vs real data).
                    # The mask (latent-space, [B, H_lat, W_lat]) is non-zero only on the crop.
                    if train_mask is not None:
                        # Vectorised per-sample bounding box from latent-space mask.
                        # mask_t: [B, H_lat, W_lat], pred_pixels: [B, 3, H_px, W_px]
                        mask_t = train_mask.unsqueeze(1).float()  # [B, 1, H_lat, W_lat] for interpolation
                        mask_px = F.interpolate(
                            mask_t,
                            size=pred_pixels.shape[-2:],
                            mode='nearest',
                        )  # [B, 1, H_px, W_px]
                        m = mask_px[:, 0]  # [B, H_px, W_px]
                        B = m.shape[0]
                        rows_any = (m > 0.5).any(dim=2)  # [B, H_px]
                        cols_any = (m > 0.5).any(dim=1)  # [B, W_px]
                        # Cumsum trick: first/last nonzero index per sample
                        rows_cs = rows_any.cumsum(dim=1)
                        rows_cs_rev = rows_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                        cols_cs = cols_any.cumsum(dim=1)
                        cols_cs_rev = cols_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                        has_mask = rows_any.any(dim=1) & cols_any.any(dim=1)
                        y1 = ((rows_cs == 1) & rows_any).int().argmax(dim=1)
                        y2 = ((rows_cs_rev == 1) & rows_any).int().argmax(dim=1) + 1
                        x1 = ((cols_cs == 1) & cols_any).int().argmax(dim=1)
                        x2 = ((cols_cs_rev == 1) & cols_any).int().argmax(dim=1) + 1
                        no_mask = ~has_mask
                        if no_mask.any():
                            y1[no_mask] = 0
                            y2[no_mask] = pred_pixels.shape[2]
                            x1[no_mask] = 0
                            x2[no_mask] = pred_pixels.shape[3]
                        fd_pixels = torch.stack([
                            pred_pixels[b, :, y1[b]:y2[b], x1[b]:x2[b]]
                            for b in range(B)
                        ], dim=0)
                    else:
                        # Full-res mode: use the whole image as-is
                        fd_pixels = pred_pixels

                    # Compute FD-Loss (gradient-preserving) on crop-only pixels
                    fd_loss, fd_dict = fd_manager.compute_loss(fd_pixels)

                    # Add FD-Loss to total loss (weighted)
                    fd_weight = getattr(t, 'fd_loss_weight', 0.1)
                    loss = loss + fd_weight * fd_loss

                    # Enqueue features for next step (detached, no grad) — crop-only
                    # source_type=1 marks these as machine-generated
                    # During warmup (pbar.n < fd_warmup_steps), we skip enqueuing
                    # generated features to prevent early noisy outputs from
                    # polluting the reference distribution.  FD-Loss is still
                    # computed against the pre-filled real data.
                    if fd_warmup_steps <= 0 or pbar.n >= fd_warmup_steps:
                        fd_manager.enqueue_features(fd_pixels, source_images=fd_pixels, source_type=1)
                    elif pbar.n == 0:
                        t.a.print(f"  FD warmup: skipping generated feature enqueue for first {fd_warmup_steps} steps")

                    # Log FID values
                    _fd_str = ", ".join([f"{k}={v:.2f}" for k, v in fd_dict.items()])

                    # Periodic cluster logging for inter-epoch inspection
                    if _fd_cluster_step > 0 and pbar.n > 0 and pbar.n % _fd_cluster_step == 0:
                        try:
                            _html = fd_manager.get_cluster_summary_html()
                            t.a.print(f"\n[Step {pbar.n}] FD queue clusters:\n{_html}")
                        except Exception:
                            pass
                else:
                    _fd_str = ""
                # -----------------------------------------------------------------

                # --- Pause-and-inspect: wait for user guidance at configurable intervals ---
                if _fd_pause_interval > 0 and pbar.n > 0 and pbar.n % _fd_pause_interval == 0:
                    _fd_pause_step = pbar.n
                    _fd_paused = True
                    _fd_resume_signal = False
                    t.a.print(f"\n[Step {pbar.n}] ⏸️  Training paused for cluster inspection. "
                              "Switch to the 'FD Cluster Inspector' tab, review clusters, "
                              "then click 'Resume Training' to continue.")
                    # Busy-wait until the UI signals resume
                    import time as _time
                    while _fd_paused and not _fd_resume_signal:
                        _time.sleep(0.5)
                        # Also check if user requested a full stop
                        if stoptimer > 0:
                            break
                    _fd_last_pause_step = _fd_pause_step
                    _fd_paused = False
                    _fd_pause_step = 0
                    t.a.print(f"[Step {pbar.n}] ▶️  Resuming training.")
                # -----------------------------------------------------------------

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                _mode_tag = f"/{t.hybrid_processing_mode}" if _train_hybrid and hasattr(t, 'hybrid_processing_mode') else ""
                _fd_tag = f" FD: {_fd_str}" if _fd_str else ""
                pbar.set_description(
                    f"Loss EMA * 1000: {loss_ema * 1000:.4f}, LR: " + ", ".join(c_lrs) +
                    f", TS: {ts_lo}-{ts_hi}{_mode_tag}{_fd_tag}, Epoch: {t.dataloader.epoch}"
                )
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(t, pbar.n, loss_ema,
                            [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lr_scheduler.get_last_lr()],
                            t.csvpath)

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del model_pred
                flush()

                # Update hybrid mode for the NEXT batch fetch (1-step ahead)
                _set_hybrid_mode(pbar.n / max(1, t.train_iterations - 1))

                result = finisher(network, t, pbar.n)
                if result is not None:
                    return result

            if pbar.n >= t.train_iterations:
                break

    return savecount(network, t, 0)


def train_diff2(t):
    global stoptimer
    global _fd_pause_interval, _fd_paused, _fd_resume_signal, _fd_pause_step, _fd_last_pause_step
    stoptimer = 0

    if t.mode == "ADDifT":
        t.orig_latent = image2latent(t, t.images[0]).to(t.train_model_precision)
        t.targ_latent = image2latent(t, t.images[1]).to(t.train_model_precision)
        data = dataset.LatentsConds(t, [([t.orig_latent, None, t.orig_cond, None],
                                         [t.targ_latent, None, t.targ_cond, None])])
        dataloaders = [dataset.DataLoader(data, batch_size=t.train_batch_size, shuffle=True)]
    else:
        t.a.print("Preparing image latents and text-conditional...")
        dataloaders = dataset.make_dataloaders(t)

    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = t.a.prepare(t.dataloader)

    t.a.print("Train Anima Multi-ADDifT Start")

    if not t.dataloader.data:
        return "No data!"

    # --- FD-Loss setup -------------------------------------------------------
    _use_fd_loss = getattr(t, 'fd_loss_enable', False)
    fd_manager = None
    if _use_fd_loss:
        t.a.print("Initializing FD-Loss...")
        fd_repr_models = getattr(t, 'fd_repr_models', 'dinov2_vitb14').strip()
        fd_repr_models = [m.strip() for m in fd_repr_models.split(",") if m.strip()]
        fd_queue_size = getattr(t, 'fd_queue_size', 50000)
        fd_queue_mode = getattr(t, 'fd_queue_mode', 'online_accum')
        fd_ema_beta = getattr(t, 'fd_ema_beta', 0.9999)
        fd_weight = getattr(t, 'fd_loss_weight', 0.1)
        fd_fid_norm_eps = getattr(t, 'fd_fid_norm_eps', 1e-6)
        # --- New: intelligent queue management parameters ---
        fd_eviction_mode = getattr(t, 'fd_eviction_mode', 'fifo')
        fd_guidance_strength = getattr(t, 'fd_guidance_strength', 0.5)
        fd_n_clusters = getattr(t, 'fd_n_clusters', 20)
        fd_cluster_log_interval = getattr(t, 'fd_cluster_log_interval', 0)  # 0 = disabled
        fd_store_source_images = getattr(t, 'fd_store_source_images', False)
        fd_enqueue_generated = getattr(t, 'fd_enqueue_generated', True)
        fd_pause_interval = getattr(t, 'fd_pause_interval', 0)  # 0 = disabled
        # Warmup: skip enqueuing generated features for the first N training steps.
        fd_warmup_steps = getattr(t, 'fd_warmup_steps', 0)  # 0 = disabled
        # ----------------------------------------------------
        # Set pause interval on module-level variable for UI access
        _fd_pause_interval = fd_pause_interval

        fd_manager = FDLossManager(
            repr_models=fd_repr_models,
            queue_size=fd_queue_size,
            queue_mode=fd_queue_mode,
            ema_beta=fd_ema_beta,
            weights=[fd_weight] * len(fd_repr_models),
            fid_norm_eps=fd_fid_norm_eps,
            device=CUDA,
            # --- New ---
            eviction_mode=fd_eviction_mode,
            guidance_strength=fd_guidance_strength,
            n_clusters=fd_n_clusters,
            store_source_images=fd_store_source_images,
            enqueue_generated=fd_enqueue_generated,
        )
        t.a.print(f"  FD-Loss judges: {fd_repr_models}, queue_size={fd_queue_size}, mode={fd_queue_mode}, eviction={fd_eviction_mode}")

        # Pre-fill queues with real data so the covariance estimate is
        # non-degenerate from step 1 (avoids eigendecomposition failures).
        # In texture mode, pass mask_key="mask" so only the crop region is
        # enqueued — the frequency-matched noise background is excluded.
        _pf_mask_key = "mask" if getattr(t, 'texture_mode', False) else None
        t.a.print("  Pre-filling FD-Loss queues with training data...")
        fd_manager.prefill_from_dataloader(t.dataloader, t.vae, mask_key=_pf_mask_key)
        t.a.print("  FD-Loss queues pre-filled.")

        # Log initial cluster structure if interval is set
        _fd_cluster_step = fd_cluster_log_interval
        if _fd_cluster_step > 0:
            try:
                _html = fd_manager.get_cluster_summary_html()
                t.a.print(f"  Initial queue clusters:\n{_html}")
            except Exception:
                pass
        # Expose to Gradio UI via module-level reference
        global _fd_manager
        _fd_manager = fd_manager
    # -------------------------------------------------------------------------

    if not getattr(t, 'texture_mode', False) and not _use_fd_loss:
        del t.vae
        if "BASE" not in t.network_blocks:
            del t.text_model

    flush()

    network, optimizer, lr_scheduler = create_network(t)

    loss_ema = None
    noise = None
    loss_velocity = None

    ts_range_lo = max(0, t.train_min_timesteps)
    ts_range_hi = max(ts_range_lo + 1, min(1000, t.train_max_timesteps))
    ts_span = ts_range_hi - ts_range_lo
    num_bands = max(1, ts_span // 100)

    time_min = ts_range_lo
    time_max = ts_range_hi

    pbar = tqdm(range(t.train_iterations))
    epoch = 0
    while t.train_iterations >= pbar.n + 1:
        for batch in t.dataloader:
            orig_latent = batch["orig_latent"]
            targ_latent = batch["targ_latent"]

            batch_size = orig_latent.shape[0]

            # For Multi-ADDifT the dataset provides per-pair conditioning from
            # each image's caption; use it when present so the model sees the
            # actual semantic difference, not identical trigger-word embeddings.
            if "orig_cond1" in batch and batch["orig_cond1"] is not None:
                orig_conds1 = batch["orig_cond1"]
                targ_conds1 = batch.get("targ_cond1", orig_conds1)
            else:
                orig_conds1 = expand_cond(t.orig_cond, batch_size)
                targ_conds1 = expand_cond(t.targ_cond, batch_size)

            orig_conds1 = move_cond_to_device(orig_conds1, CUDA, t.train_model_precision)
            targ_conds1 = move_cond_to_device(targ_conds1, CUDA, t.train_model_precision)

            optimizer.zero_grad()
            noise = torch.randn_like(orig_latent)

            turn = pbar.n % 2 == 0

            if turn:
                band_span = ts_span // num_bands
                index = (pbar.n // 2) % num_bands
                time_min = ts_range_lo + band_span * index
                time_max = time_min + band_span
                time_max = max(time_min + 1, min(time_max, ts_range_hi))

            dist_type   = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
            dist_params = getattr(t, 'train_ts_dist_params', '') or ''
            n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
            band_lo = int(min(time_min, ts_range_hi - 1))
            band_hi = int(max(time_max, ts_range_lo + 1))
            timesteps = _sample_timesteps(band_lo, band_hi, n_ts, CUDA, dist_type, dist_params)
            timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

            orig_noisy_latents = t.noise_scheduler.add_noise(
                orig_latent if turn else targ_latent, noise, timesteps
            )
            targ_noisy_latents = t.noise_scheduler.add_noise(
                targ_latent if turn else orig_latent, noise, timesteps
            )

            orig_noisy_latents = orig_noisy_latents.to(CUDA, dtype=t.train_model_precision)
            targ_noisy_latents = targ_noisy_latents.to(CUDA, dtype=t.train_model_precision)

            # Baseline: base model prediction (no LoRA) on source latent.
            # multiplier must be 0 here — LoRANetwork starts at 1, so set explicitly.
            network.set_multiplier(0)
            with torch.no_grad(), t.a.autocast():
                orig_noise_pred = anima_forward(t, orig_noisy_latents, timesteps, orig_conds1)

            # LoRA-modified prediction on target latent, alternating direction each step.
            network.set_multiplier(0.25 if turn else -0.25)
            with t.a.autocast():
                targ_noise_pred = anima_forward(t, targ_noisy_latents, timesteps, targ_conds1)

            network.set_multiplier(0)

            if t.diff_use_diff_mask and "mask" in batch:
                mask = F.interpolate(batch["mask"].to(CUDA).unsqueeze(1).float(), size=targ_noise_pred.shape[2:], mode='nearest')
                targ_noise_pred = targ_noise_pred * mask
                orig_noise_pred = orig_noise_pred * mask

            loss, loss_ema, loss_velocity = process_loss(
                t, targ_noise_pred, orig_noise_pred, timesteps, loss_ema, loss_velocity
            )

            # --- FD-Loss: perceptual quality via differentiable FID ----------
            if _use_fd_loss and fd_manager is not None:
                # Reconstruct predicted clean latents from the LoRA-modified prediction
                # Flow matching: noisy = (1-t)*clean + t*noise
                # velocity = noise - clean
                # pred_clean = noisy - ts * velocity
                ts_norm = timesteps.float() / 1000.0
                # Use the target (LoRA-modified) prediction for FD-Loss evaluation
                pred_clean = targ_noisy_latents.float() - ts_norm.view(-1, 1, 1, 1) * targ_noise_pred.float()

                # Decode predicted clean latents to pixels [0, 1]
                pred_pixels = latent2pixels(t, pred_clean)

                # In texture mode, the batch contains a full canvas with frequency-matched
                # noise background + a placed crop.  Feed ONLY the crop region to FD-Loss.
                # The mask (latent-space, [B, H_lat, W_lat]) is non-zero only on the crop.
                # Vectorised per-sample bounding box via cumsum trick (no Python loop).
                if "mask" in batch and batch["mask"] is not None:
                    mask_t = batch["mask"].to(CUDA)  # [B, H_lat, W_lat]
                    mask_px = F.interpolate(
                        mask_t.unsqueeze(1).float(),  # [B, 1, H_lat, W_lat]
                        size=pred_pixels.shape[-2:],
                        mode='nearest',
                    )  # [B, 1, H_px, W_px]
                    m = mask_px[:, 0]  # [B, H_px, W_px]
                    B = m.shape[0]
                    rows_any = (m > 0.5).any(dim=2)  # [B, H_px]
                    cols_any = (m > 0.5).any(dim=1)  # [B, W_px]
                    rows_cs = rows_any.cumsum(dim=1)
                    rows_cs_rev = rows_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                    cols_cs = cols_any.cumsum(dim=1)
                    cols_cs_rev = cols_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                    has_mask = rows_any.any(dim=1) & cols_any.any(dim=1)
                    y1 = ((rows_cs == 1) & rows_any).int().argmax(dim=1)
                    y2 = ((rows_cs_rev == 1) & rows_any).int().argmax(dim=1) + 1
                    x1 = ((cols_cs == 1) & cols_any).int().argmax(dim=1)
                    x2 = ((cols_cs_rev == 1) & cols_any).int().argmax(dim=1) + 1
                    no_mask = ~has_mask
                    if no_mask.any():
                        y1[no_mask] = 0
                        y2[no_mask] = pred_pixels.shape[2]
                        x1[no_mask] = 0
                        x2[no_mask] = pred_pixels.shape[3]
                    fd_pixels = torch.stack([
                        pred_pixels[b, :, y1[b]:y2[b], x1[b]:x2[b]]
                        for b in range(B)
                    ], dim=0)
                else:
                    fd_pixels = pred_pixels

                # Compute FD-Loss (gradient-preserving) on crop-only pixels
                fd_loss, fd_dict = fd_manager.compute_loss(fd_pixels)

                # Add FD-Loss to total loss (weighted)
                fd_weight = getattr(t, 'fd_loss_weight', 0.1)
                loss = loss + fd_weight * fd_loss

                # Enqueue features for next step (detached, no grad) — crop-only
                # source_type=1 marks these as machine-generated
                # During warmup (pbar.n < fd_warmup_steps), skip enqueuing
                # generated features to prevent early noisy outputs from
                # polluting the reference distribution.
                if fd_warmup_steps <= 0 or pbar.n >= fd_warmup_steps:
                    fd_manager.enqueue_features(fd_pixels, source_images=fd_pixels, source_type=1)
                elif pbar.n == 0:
                    t.a.print(f"  FD warmup: skipping generated feature enqueue for first {fd_warmup_steps} steps")

                _fd_str = ", ".join([f"{k}={v:.2f}" for k, v in fd_dict.items()])
            else:
                _fd_str = ""
            # -----------------------------------------------------------------

            # --- Pause-and-inspect: wait for user guidance at configurable intervals ---
            if _fd_pause_interval > 0 and pbar.n > 0 and pbar.n % _fd_pause_interval == 0:
                _fd_pause_step = pbar.n
                _fd_paused = True
                _fd_resume_signal = False
                t.a.print(f"\n[Step {pbar.n}] ⏸️  Training paused for cluster inspection. "
                          "Switch to the 'FD Cluster Inspector' tab, review clusters, "
                          "then click 'Resume Training' to continue.")
                # Busy-wait until the UI signals resume
                import time as _time
                while _fd_paused and not _fd_resume_signal:
                    _time.sleep(0.5)
                    # Also check if user requested a full stop
                    if stoptimer > 0:
                        break
                _fd_last_pause_step = _fd_pause_step
                _fd_paused = False
                _fd_pause_step = 0
                t.a.print(f"[Step {pbar.n}] ▶️  Resuming training.")
            # -----------------------------------------------------------------

            c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
            _fd_tag = f" FD: {_fd_str}" if _fd_str else ""
            pbar.set_description(
                f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velocity: {loss_velocity * 1000:.4f}, "
                f"Current LR: " + ", ".join(c_lrs) + f", Epoch: {epoch}{_fd_tag}"
            )
            pbar.update(1)

            if t.logging_save_csv:
                savecsv(t, pbar.n, loss_ema,
                        [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lr_scheduler.get_last_lr()],
                        t.csvpath)

            t.a.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            flush()

            result = finisher(network, t, pbar.n)
            if result is not None:
                del optimizer, lr_scheduler
                return result

        epoch += 1

    return savecount(network, t, 0)


# --------------------------------------------------------------------------- #
# Network / optimizer / scheduler helpers                                      #
# --------------------------------------------------------------------------- #

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def _sample_timesteps(ts_lo: int, ts_hi: int, n: int, device,
                       dist_type: str = "flow_shift", dist_params: str = "") -> torch.Tensor:
    """Sample n integer timesteps in [ts_lo, ts_hi) using the specified distribution.

    dist_type:
      "uniform"      — flat torch.randint
      "flow_shift"   — bias toward high-noise via shift factor (sd-scripts convention)
      "logit_normal" — sigmoid of Normal(mean, std); params: mean=0.0 std=1.0
      "cosmap"       — cosine bijection; bias toward mid-noise
      "beta"         — Beta(alpha, beta); params: alpha=0.5 beta=0.5
                       alpha=beta=0.5 → inverse bell (U-shape), alpha=beta>1 → bell,
                       alpha=beta=1 → uniform, alpha≠beta → skewed
    """
    import math as _math

    span = ts_hi - ts_lo

    # Parse "key=value ..." pairs shared by logit_normal and beta
    params: dict = {}
    for item in (dist_params or "").replace(",", " ").split():
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                params[k.strip()] = float(v.strip())
            except ValueError:
                pass

    if dist_type == "uniform":
        return torch.randint(ts_lo, ts_hi, (n,), device=device)

    if dist_type == "logit_normal":
        mean = params.get("mean", 0.0)
        std = max(0.01, params.get("std", 1.0))
        u = torch.randn(n, device=device) * std + mean
        sigma = torch.sigmoid(u)
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    if dist_type == "cosmap":
        # sigma = 1 - 1 / (tan(pi/2 * u) + 1)  for u ~ Uniform(0, 1)
        u = torch.rand(n, device=device).clamp(1e-6, 1.0 - 1e-6)
        sigma = 1.0 - 1.0 / (torch.tan(u * (_math.pi / 2.0)) + 1.0)
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    if dist_type == "beta":
        # Beta(alpha, beta): alpha=beta=0.5 → inverse bell, alpha=beta>1 → bell
        alpha = max(0.01, params.get("alpha", 0.5))
        beta  = max(0.01, params.get("beta",  0.5))
        sigma = torch.distributions.Beta(
            torch.tensor(alpha, dtype=torch.float32, device=device),
            torch.tensor(beta,  dtype=torch.float32, device=device),
        ).sample((n,))
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    # Default: "flow_shift" — shift parsed from dist_params
    shift = max(0.0, params.get("shift", 3.0))
    if shift <= 1.0:
        return torch.randint(ts_lo, ts_hi, (n,), device=device)
    u = torch.rand(n, device=device)
    sigma = (u * shift) / (1.0 + (shift - 1.0) * u)
    return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)


# --------------------------------------------------------------------------- #
# Timestep schedule                                                            #
# --------------------------------------------------------------------------- #

def _parse_ts_schedule_text(text: str):
    """Parse a timestep schedule from inline text (stored directly in the config).

    Format — one entry per non-comment line::

        # Columns: step_pct   t_min   t_max   [mode]   [lr]
        #
        # step_pct  0.0–1.0, fraction of total training steps at which row activates
        # mode      texture | fullres | - (- or blank = keep current mode)
        # lr        effective learning rate for this phase (e.g. 1e-4); omit to keep scheduler value
        #
        # Weighting is always flat (uniform).
        # Step-function: the LAST row whose step_pct ≤ current fraction is active.
        # Example — curriculum that widens range, switches mode, and adjusts LR:
        #
        # 0.00   200   800   fullres   1e-4
        # 0.40     0   700   texture   5e-5
        # 0.70     0  1000   texture

    Returns a sorted list of (step_pct, ts_lo, ts_hi, mode, lr_override) tuples,
    or None if text is empty / contains no valid entries.
    lr_override is a float or None if not specified.
    """
    entries = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue  # ignore legacy section headers gracefully
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            mode = parts[3].lower() if len(parts) > 3 else ""
            if mode == "-":
                mode = ""
            lr_override = float(parts[4]) if len(parts) > 4 else None
            entry = (
                float(parts[0]),
                int(parts[1]),
                int(parts[2]),
                mode,
                lr_override,
            )
        except (ValueError, IndexError):
            continue
        entries.append(entry)

    if not entries:
        return None
    entries.sort(key=lambda e: e[0])
    return entries   # list of (step_pct, ts_lo, ts_hi, mode, lr_override)


def _resolve_ts_entry(entries: list, step_pct: float):
    """Step-function lookup: return the last entry whose step_pct ≤ current."""
    result = entries[0]
    for e in entries:
        if e[0] <= step_pct:
            result = e
        else:
            break
    return result  # (step_pct, ts_lo, ts_hi, mode, lr_override)


def create_network(t):
    network = load_network(t)
    optimizer = trainer.get_optimizer(
        t.train_optimizer,
        network.prepare_optimizer_params(),
        t.train_learning_rate,
        t.train_optimizer_settings,
        network,
    )

    t.is_schedulefree = t.train_optimizer.endswith("schedulefree".lower())

    if t.is_schedulefree:
        optimizer.train()
    else:
        lr_scheduler = trainer.load_lr_scheduler(t, optimizer)

    print(f"Optimizer : {type(optimizer).__name__}")
    print(f"Optimizer Settings : {t.train_optimizer_settings}")

    network, optimizer, lr_scheduler = t.a.prepare(
        network, optimizer, None if t.is_schedulefree else lr_scheduler
    )

    return network, optimizer, DummyScheduler(optimizer) if t.is_schedulefree else lr_scheduler


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_last_lr(self):
        return [p["scheduled_lr"] for p in self.optimizer.param_groups]

    def step(self):
        pass


def load_network(t):
    # Anima DiT uses standard linear LoRA (lierla) — no convolutions, no loha
    return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)


def stop_time(save):
    global stoptimer
    stoptimer = 2 if save else 1


def finisher(network, t, i, copy=False):
    if t.save_list and i >= t.save_list[0]:
        savecount(network, t, t.save_list.pop(0), copy)

    if stoptimer > 0:
        if stoptimer > 1:
            result = ". " + savecount(network, t, i, copy)
        else:
            result = ""
        return "Stopped" + result


def savecount(network, t, i, copy=False):
    if t.metadata == {}:
        metadator(t)
    if copy and False:  # diff_save_1st_pass removed (DDPM 2-pass concept)
        return "Not save copy"
    add = "_copy" if copy else ""
    add = f"{add}_{i}steps" if i > 0 else add
    filename = os.path.join(t.save_dir, f"{t.save_lora_name}{add}.safetensors")
    print(f" Saving to {filename}")
    metaname = f"{t.save_lora_name}{add}"
    filename = network.save_weights(filename, t, metaname)
    return f"Successfully created to {filename}"


def makesavelist(t):
    if t.save_per_steps > 0:
        t.save_list = [x * t.save_per_steps for x in range(1, t.train_iterations // t.save_per_steps + 1)]
        if t.train_iterations in t.save_list:
            t.save_list.remove(t.train_iterations)
    else:
        t.save_list = []


def process_loss(t, original, target, timesteps, loss_ema, loss_velocity,
                 mask=None, copy=False, ts_weights=None):
    if t.train_loss_function == "MSE":
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "L1":
        loss = torch.nn.functional.l1_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "Smooth-L1":
        loss = torch.nn.functional.smooth_l1_loss(original.float(), target.float(), reduction="none")
    else:
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")

    if mask is not None:
        # Weighted mean over the masked region; empty canvas → zero gradient.
        # mask: [B, H, W] → [B, 1, H, W] → expand to [B, C, H, W] so the
        # denominator counts every active (channel, spatial) element.
        m = mask.to(loss.device)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        m = m.expand_as(loss)
        loss = (loss * m).sum(dim=[1, 2, 3]) / m.sum(dim=[1, 2, 3]).clamp(min=1e-8)
    else:
        loss = loss.mean([1, 2, 3])   # [B]

    # Per-timestep loss weighting: downweight artifact-prone timestep extremes.
    # ts_weights is [B] in (0, 1]; None means flat (no reweighting).
    if ts_weights is not None and ts_weights.shape[0] == loss.shape[0]:
        w = ts_weights.to(loss.device)
        loss = (loss * w).sum() / w.sum().clamp(min=1e-8)
    else:
        loss = loss.mean()

    if loss_ema is None:
        loss_ema = loss.item()
        loss_velocity = 0
    else:
        loss_velocity = loss_velocity * 0.9 + (loss_ema - (loss_ema * 0.9 + loss.item() * 0.1)) * 0.1
        loss_ema = loss_ema * 0.9 + loss.item() * 0.1

    return loss, loss_ema, loss_velocity


# --------------------------------------------------------------------------- #
# Encode helpers                                                               #
# --------------------------------------------------------------------------- #

def image2latent(t, image):
    """Encode an image (PIL or path) to Anima latents using the Anima VAE."""
    if isinstance(image, str):
        with Image.open(image) as img:
            image = img.convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")

    image_np = numpy.array(image).astype(numpy.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1] (Anima VAE expects [-1, 1])
    image_tensor = image_tensor.to(CUDA, dtype=t.train_model_precision)

    with torch.no_grad():
        latent = t.vae.encode_pixels_to_latents(image_tensor)  # [1, C, H, W]

    return latent


def latent2pixels(t, latent):
    """Decode VAE latents back to pixel space [0, 1] for FD-Loss feature extraction.

    Gradients flow through the VAE decode (VAE is frozen, so this is safe).
    The VAE's decode_to_pixels handles the latent normalization internally.

    Args:
        t: Trainer instance with .vae attribute.
        latent: Tensor [B, C, H, W] in VAE latent space.

    Returns:
        Tensor [B, 3, H*8, W*8] in [0, 1] range.
    """
    # decode_to_pixels expects [B, C, H, W] or [B, C, 1, H, W]
    # Returns [-1, 1] range. VAE is frozen so gradients pass through safely.
    pixels = t.vae.decode_to_pixels(latent.float())
    # [-1, 1] -> [0, 1] for FD-Loss feature extractors (expect [0, 1] input)
    return pixels * 0.5 + 0.5


def text2cond(t, prompt):
    """Encode a text prompt into Anima conditioning tensors."""
    cond, _ = t.text_model.encode_text(prompt if isinstance(prompt, list) else [prompt])
    return cond, None


# --------------------------------------------------------------------------- #
# Debug / logging                                                              #
# --------------------------------------------------------------------------- #

CSVHEADS = ["network_rank", "network_alpha", "train_learning_rate", "train_iterations",
            "train_lr_scheduler", "model_version", "train_optimizer", "save_lora_name"]


def savecsv(t, step, loss, lr, csvpath, copy=False):
    header = []
    for key in CSVHEADS:
        header.append([key, getattr(t, key, "")])
    header.append(["Step", "Loss"] + ["Learning Rate " + str(i + 1) for i in range(len(lr))])

    if copy:
        csvpath = csvpath.replace(".csv", "_copy.csv")

    directory = os.path.dirname(csvpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_exists = os.path.isfile(csvpath)
    with open(csvpath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            for head in header:
                writer.writerow(head)
        writer.writerow([step, loss] + lr)


def metadator(t):
    t.metadata = {
        "ss_session_id": random.randint(0, 2**32),
        "ss_training_started_at": time.time(),
        "ss_output_name": t.save_lora_name,
        "ss_learning_rate": t.train_learning_rate,
        "ss_max_train_steps": t.train_iterations,
        "ss_lr_warmup_steps": 0,
        "ss_lr_scheduler": t.train_lr_scheduler,
        "ss_network_module": "network.lora",
        "ss_network_dim": t.network_rank,
        "ss_network_alpha": t.network_alpha,
        "ss_mixed_precision": t.train_lora_precision,
        "ss_lr_step_rules": "",
        "ss_lr_scheduler_num_cycles": 1,
        "ss_lr_scheduler_power": t.train_lr_scheduler_power,
        "ss_v2": False,
        "ss_base_model_version": t.model_version,
        "ss_seed": t.train_seed,
        "ss_optimizer": t.train_optimizer,
        "ss_min_snr_gamma": 0,
        "ss_tag_frequency": json.dumps({1: t.count_dict}),
    }

# --------------------------------------------------------------------------- #
# Gradio UI accessors for the interactive cluster panel                        #
# These are called from scripts/traintrain.py while training is running.       #
# --------------------------------------------------------------------------- #

def get_fd_manager():
    """Return the active FDLossManager instance, or None if FD-Loss is off."""
    return _fd_manager


def render_cluster_panel_ui(n_clusters: Optional[int] = None,
                            max_per_cluster: int = 9) -> str:
    """Render the interactive cluster panel HTML for the Gradio UI.

    Called from the UI refresh button.  Returns an HTML string or a
    placeholder message if FD-Loss is not active.
    """
    fd = _fd_manager
    if fd is None:
        return ("<div style='color:#888;font-family:sans-serif;font-size:13px;'>"
                "FD-Loss is not enabled.  Check <b>fd_loss_enable</b> and start training.</div>")
    try:
        return fd.render_interactive_cluster_panel(
            n_clusters=n_clusters,
            max_per_cluster=max_per_cluster,
        )
    except Exception as e:
        return (f"<div style='color:#f88;font-family:sans-serif;font-size:13px;'>"
                f"Error rendering cluster panel: {e}</div>")


def fetch_thumbnail_base64(queue_idx: int) -> str:
    """Return a base64-encoded PNG data URI for a single queue index.

    Called on-demand from the JS lazy thumbnail loader when the user
    expands a cluster accordion.  Only the requested thumbnail is
    base64-encoded, avoiding the cost of encoding all thumbnails at once.
    """
    fd = _fd_manager
    if fd is None:
        return ""
    try:
        return fd.fetch_thumbnail_base64(queue_idx)
    except Exception:
        return ""


def cluster_panel_toggle_protect_cluster(cluster_id: int) -> str:
    """Stage a protect/unprotect cluster action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        # Determine current projected state using LIVE clustering.
        # Cluster-level actions are resolved to per-index actions at stage time
        # (see FDLossManager.protect_cluster), so we use live clustering here
        # to determine the correct toggle direction.
        queue = fd.judges[0]["queue"]
        projected = fd._get_projected_protected_mask(queue)
        clusters = queue.get_clusters(n_clusters=fd.n_clusters)
        if not clusters or "assignments" not in clusters:
            return fd.render_interactive_cluster_panel()
        indices = (clusters["assignments"] == cluster_id).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return fd.render_interactive_cluster_panel()
        if projected[indices].any():
            fd.unprotect_cluster(cluster_id)
        else:
            fd.protect_cluster(cluster_id)
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


def cluster_panel_set_guidance(cluster_id: int) -> str:
    """Stage a set-guidance action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        fd.set_guidance_from_cluster(cluster_id)
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


def cluster_panel_clear_guidance() -> str:
    """Stage a clear-guidance action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        fd.clear_guidance_target()
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


def cluster_panel_toggle_evict_cluster(cluster_id: int) -> str:
    """Stage a toggle-eviction cluster action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        fd.evict_cluster(cluster_id)
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


def cluster_panel_toggle_protect_index(idx: int) -> str:
    """Stage a protect/unprotect index action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        # Determine current projected state
        queue = fd.judges[0]["queue"]
        projected = fd._get_projected_protected_mask(queue)
        if projected[idx].item():
            fd.unprotect_index(idx)
        else:
            fd.protect_index(idx)
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


# --------------------------------------------------------------------------- #
# Pause-and-inspect UI accessors                                               #
# --------------------------------------------------------------------------- #

def set_fd_pause_interval(interval: int):
    """Set the pause interval (steps between pauses).  0 = disabled."""
    global _fd_pause_interval
    _fd_pause_interval = interval


def get_fd_pause_interval() -> int:
    """Return the current pause interval."""
    return _fd_pause_interval


def resume_training() -> str:
    """Signal the training loop to resume from a pause.

    Returns a status message for the UI.
    """
    global _fd_resume_signal, _fd_paused, _fd_just_resumed
    if not _fd_paused:
        return "<div style='color:#888;'>Training is not paused.</div>"
    _fd_resume_signal = True
    _fd_just_resumed = True
    return "<div style='color:#4ade80;'>▶️ Resume signal sent.</div>"


def is_training_paused() -> bool:
    """Return True if training is currently paused for inspection."""
    return _fd_paused


def get_pause_status_html() -> str:
    """Return an HTML snippet showing the current pause state."""
    global _fd_just_resumed, _fd_last_pause_step
    if _fd_just_resumed:
        _fd_just_resumed = False  # consume the flag
        return ("<div style='padding:8px 12px;background:#1a2a1a;border:1px solid #4ade80;"
                "border-radius:4px;font-family:sans-serif;font-size:13px;color:#4ade80;'>"
                "▶️ <b>Training Resumed</b> — continuing from step "
                f"<b>{_fd_last_pause_step}</b>.</div>")
    if _fd_paused:
        return ("<div style='padding:8px 12px;background:#1a2a1a;border:1px solid #4ade80;"
                "border-radius:4px;font-family:sans-serif;font-size:13px;color:#4ade80;'>"
                "⏸️ <b>Training Paused</b> at step "
                f"<b>{_fd_pause_step}</b>.  Inspect clusters, perform guidance, "
                "then click <b>Resume Training</b>.</div>")
    if _fd_pause_interval > 0:
        return ("<div style='padding:8px 12px;background:#1a1a2a;border:1px solid #60a5fa;"
                "border-radius:4px;font-family:sans-serif;font-size:13px;color:#60a5fa;'>"
                f"ℹ️ Pause-and-inspect is active (every <b>{_fd_pause_interval}</b> steps). "
                "Training will pause automatically at the next interval.</div>")
    return ("<div style='padding:8px 12px;background:#1a1a1a;border:1px solid #555;"
            "border-radius:4px;font-family:sans-serif;font-size:13px;color:#888;'>"
            "Pause-and-inspect is disabled.  Set <b>fd_pause_interval</b> > 0 to enable.</div>")


def cluster_panel_toggle_evict_index(idx: int) -> str:
    """Stage a toggle-eviction index action.  Returns updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        fd.evict_index(idx)
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"


def cluster_panel_commit_all() -> str:
    """Execute all staged actions (protect/guidance/evict) and return updated panel HTML."""
    fd = _fd_manager
    if fd is None:
        return "<div style='color:#888;'>FD-Loss not active</div>"
    try:
        n = fd.commit_pending_actions()
        return fd.render_interactive_cluster_panel()
    except Exception as e:
        return f"<div style='color:#f88;'>Error: {e}</div>"
