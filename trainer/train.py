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
    anima_forward_refcn,
    expand_cond,
    move_cond_to_device,
)
from pprint import pprint
from accelerate.utils import set_seed

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
ML_REFCN = "RefCN"

# How often (in steps) to compute the conditioning-influence diagnostic in RefCN mode.
# Two extra forward passes (real ref + zero ref) are run with no_grad, so the cost
# is small relative to a training step.
REFCN_VAL_EVERY = 50

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
    if not args[0]:
        setcurrentname(args)
    result = train_main(*args)
    while len(queue_list) > 0:
        settings = queue_list.pop(0)
        result += "\n" + train_main(*settings)
    return result


def train_main(jsononly, mode, modelname, vaename, *args):
    t = trainer.Trainer(jsononly, modelname, vaename, mode, args)

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
        elif t.mode == ML_REFCN:
            result = train_refcn(t)
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
    # VAE must stay alive for JIT encoding: texture_mode OR hybrid mode
    if not getattr(t, 'texture_mode', False) and not _train_hybrid:
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

                # Resolve timestep range, LR, and distribution from schedule (or static config)
                step_pct = pbar.n / max(1, t.train_iterations - 1)
                if _ts_schedule is not None:
                    _entry = _resolve_ts_entry(_ts_schedule, step_pct)
                    _ts_lo_e, _ts_hi_e, _lr_override = _entry[1], _entry[2], _entry[4]
                    # None means "not specified in this entry → inherit global"
                    ts_lo = _ts_lo_e if _ts_lo_e is not None else t.train_min_timesteps
                    ts_hi = _ts_hi_e if _ts_hi_e is not None else t.train_max_timesteps
                    if _lr_override is not None:
                        for pg in optimizer.param_groups:
                            pg['lr'] = _lr_override
                    _sched_dist = _entry[5]   # dist_type override (None = inherit)
                    if _sched_dist is not None:
                        dist_type   = _sched_dist
                        dist_params = (_entry[6] or '') if _entry[6] is not None else ''
                    else:
                        dist_type   = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                        dist_params = getattr(t, 'train_ts_dist_params', '') or ''
                else:
                    ts_lo = t.train_min_timesteps
                    ts_hi = t.train_max_timesteps
                    dist_type   = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                    dist_params = getattr(t, 'train_ts_dist_params', '') or ''
                ts_lo = max(0, ts_lo)
                ts_hi = max(ts_lo + 1, min(1000, ts_hi))
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

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                _mode_tag = f"/{t.hybrid_processing_mode}" if _train_hybrid and hasattr(t, 'hybrid_processing_mode') else ""
                pbar.set_description(
                    f"Loss EMA * 1000: {loss_ema * 1000:.4f}, LR: " + ", ".join(c_lrs) +
                    f", TS: {ts_lo}-{ts_hi}{_mode_tag}, Epoch: {t.dataloader.epoch}"
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


def train_refcn(t):
    """Reference ControlNet LoRA training loop.

    Like train_lora but each batch includes a paired clean reference latent
    that is passed to the Anima DiT via the ref_latents kwarg.  The model
    learns to denoise the target conditioned on the reference, training the
    LoRA modules to attend to the additional temporal context frame.

    Dataset convention:
      - Target images:    t.lora_data_directory  (same as LoRA)
      - Reference images: t.refcn_ref_dir         (paired by filename stem)
      - If refcn_ref_dir is empty: target image is its own reference (self-ref)
    """
    global stoptimer
    stoptimer = 0

    t.a.print("Preparing paired target / reference latents...")
    dataloaders = dataset.make_dataloaders_refcn(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = t.a.prepare(t.dataloader)

    t.a.print("Train Anima Reference ControlNet LoRA Start")

    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    # All latents are pre-encoded in make_dataloaders_refcn — free VAE and
    # text model so they don't occupy VRAM during the training loop.
    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    _ts_schedule = _parse_ts_schedule_text(getattr(t, 'train_ts_schedule', '') or '')
    if _ts_schedule:
        print(f"Timestep schedule loaded: {len(_ts_schedule)} entries")

    loss_ema = None
    loss_velocity = None

    # Conditioning-influence diagnostic state.
    # Snapshotted from the first batch; reused every REFCN_VAL_EVERY steps.
    _val_snap       = None   # (noisy_latents, ref_latents, timesteps, velocity_target, conds1)
    _cond_influence = 0.0    # RMS shift caused by reference: ||pred_ref - pred_null|| / √numel
    _cond_gain      = 0.0    # (loss_null - loss_ref) × 1000 — positive = ref helps

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents     = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                ref_latents = batch["ref_latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1      = batch["cond1"] if "cond1" in batch else None

                noise      = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Timestep sampling
                step_pct = pbar.n / max(1, t.train_iterations - 1)
                if _ts_schedule is not None:
                    _entry = _resolve_ts_entry(_ts_schedule, step_pct)
                    ts_lo      = _entry[1] if _entry[1] is not None else t.train_min_timesteps
                    ts_hi      = _entry[2] if _entry[2] is not None else t.train_max_timesteps
                    _lr_override = _entry[4]
                    if _lr_override is not None:
                        for pg in optimizer.param_groups:
                            pg['lr'] = _lr_override
                    dist_type   = _entry[5] or getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                    dist_params = (_entry[6] or '') if _entry[6] is not None else getattr(t, 'train_ts_dist_params', '') or ''
                else:
                    ts_lo       = t.train_min_timesteps
                    ts_hi       = t.train_max_timesteps
                    dist_type   = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                    dist_params = getattr(t, 'train_ts_dist_params', '') or ''

                ts_lo = max(0, ts_lo)
                ts_hi = max(ts_lo + 1, min(1000, ts_hi))
                n_ts  = 1 if t.train_fixed_timsteps_in_batch else batch_size
                timesteps = _sample_timesteps(ts_lo, ts_hi, n_ts, CUDA, dist_type, dist_params)
                timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

                # Add noise to target only; reference stays clean
                noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)

                # Resolve text conditioning
                if conds1 is None:
                    conds1 = expand_cond(t.orig_cond, batch_size)
                elif isinstance(conds1, str) or (isinstance(conds1, list) and isinstance(conds1[0], str)):
                    conds1, _ = t.text_model.encode_text(conds1 if isinstance(conds1, list) else [conds1])
                elif isinstance(conds1, (tuple, list)):
                    conds1 = move_cond_to_device(conds1, CUDA, t.train_lora_precision)

                with network, t.a.autocast():
                    model_pred = anima_forward_refcn(t, noisy_latents, ref_latents, timesteps, conds1)

                velocity_target = (noise - latents).to(torch.float32)

                loss, loss_ema, loss_velocity = process_loss(
                    t, model_pred, velocity_target, timesteps, loss_ema, loss_velocity,
                )

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del model_pred
                flush()

                # ----------------------------------------------------------
                # Snapshot the first batch for the conditioning-influence
                # diagnostic.  Take only sample 0 (batch_size=1 slice) so the
                # val forward passes are cheap regardless of training batch size.
                # ----------------------------------------------------------
                if _val_snap is None:
                    _val_snap = (
                        noisy_latents[:1].detach().clone(),
                        ref_latents[:1].detach().clone(),
                        timesteps[:1].detach().clone(),
                        velocity_target[:1].detach().clone(),
                        _clone_cond(conds1),
                    )

                # ----------------------------------------------------------
                # Conditioning-influence diagnostic (every REFCN_VAL_EVERY steps).
                #
                # Runs two no-grad forward passes on the fixed val snapshot:
                #   pred_ref:  model sees the real reference latent
                #   pred_null: model sees an all-zero reference (no conditioning)
                #
                # _cond_influence: RMS shift = ||pred_ref - pred_null|| / √numel
                #   • Near 0 at init (LoRA weights ≈ 0, reference has no effect)
                #   • Should rise as the LoRA learns to use the reference
                #
                # _cond_gain: (loss_null − loss_ref) × 1000
                #   • Positive  → reference actively improves denoising
                #   • Near zero → reference changes predictions but doesn't help
                #   • Negative  → reference is hurting (misconfiguration or overfit)
                # ----------------------------------------------------------
                if pbar.n > 0 and pbar.n % REFCN_VAL_EVERY == 0 and _val_snap is not None:
                    _vn, _vr, _vts, _vtgt, _vcond = _val_snap
                    _vn   = _vn.to(CUDA,  dtype=t.train_lora_precision)
                    _vr   = _vr.to(CUDA,  dtype=t.train_lora_precision)
                    _vts  = _vts.to(CUDA)
                    _vtgt = _vtgt.to(CUDA, dtype=torch.float32)
                    if isinstance(_vcond, (tuple, list)):
                        _vcond_dev = move_cond_to_device(_vcond, CUDA, t.train_lora_precision)
                    else:
                        _vcond_dev = _vcond
                    _zero_ref = torch.zeros_like(_vr)
                    with torch.no_grad(), t.a.autocast(), network:
                        _p_ref  = anima_forward_refcn(t, _vn, _vr,       _vts, _vcond_dev).float()
                        _p_null = anima_forward_refcn(t, _vn, _zero_ref, _vts, _vcond_dev).float()
                    _null_loss = F.mse_loss(_p_null, _vtgt).item()
                    _ref_loss  = F.mse_loss(_p_ref,  _vtgt).item()
                    _cond_influence = (_p_ref - _p_null).norm().item() / (_p_ref.numel() ** 0.5)
                    _cond_gain      = (_null_loss - _ref_loss) * 1000
                    del _p_ref, _p_null, _vn, _vr, _vts, _vtgt, _vcond_dev, _zero_ref
                    flush()

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                pbar.set_description(
                    f"[RefCN] Loss: {loss_ema * 1000:.3f}"
                    f" | Cond: {_cond_influence:.4f} ({_cond_gain:+.2f})"
                    f" | LR: " + ", ".join(c_lrs) +
                    f" | TS: {ts_lo}-{ts_hi} | Epoch: {t.dataloader.epoch}"
                )
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(t, pbar.n, loss_ema,
                            [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lr_scheduler.get_last_lr()],
                            t.csvpath,
                            extra_cols={"Cond Influence": _cond_influence, "Cond Gain": _cond_gain})

                result = finisher(network, t, pbar.n)
                if result is not None:
                    return result

            if pbar.n >= t.train_iterations:
                break

    return savecount(network, t, 0)


def train_diff2(t):
    global stoptimer
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

    if not getattr(t, 'texture_mode', False):
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

            c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
            pbar.set_description(
                f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velocity: {loss_velocity * 1000:.4f}, "
                f"Current LR: " + ", ".join(c_lrs) + f", Epoch: {epoch}"
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

_flush_call_count = 0

def flush():
    global _flush_call_count
    torch.cuda.empty_cache()
    _flush_call_count += 1
    if _flush_call_count % 20 == 0:
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

def _split_schedule_tokens(s: str) -> list:
    """Split *s* on commas that are not inside parentheses."""
    tokens, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1; cur.append(ch)
        elif ch == ')':
            depth -= 1; cur.append(ch)
        elif ch == ',' and depth == 0:
            t = ''.join(cur).strip()
            if t:
                tokens.append(t)
            cur = []
        else:
            cur.append(ch)
    t = ''.join(cur).strip()
    if t:
        tokens.append(t)
    return tokens


def _parse_new_schedule_line(line: str):
    """Parse a ``@<step_pct>[, key=value ...]`` schedule line.

    Recognised keys
    ---------------
    range=<lo>-<hi>
        Timestep window, e.g. ``range=50-450``.
    ts_dist=<name>[(<params>)]
        Distribution and its parameters, e.g.
        ``ts_dist=flow_shift(shift=3.0)``  or  ``ts_dist=beta(alpha=0.14,beta=1.0)``.
        Commas inside the parentheses are preserved and forwarded to *_sample_timesteps*.
    mode=<texture|fullres|-|...>  **or bare word** ``texture`` / ``fullres``
        Processing mode for hybrid training.
    lr=<float>
        Override learning rate for this phase, e.g. ``lr=1e-4``.

    Returns a 7-tuple ``(step_pct, ts_lo, ts_hi, mode, lr_override, dist_type, dist_params)``
    where any field can be *None* meaning "inherit the global setting".
    Returns *None* on parse failure.
    """
    import re as _re
    assert line.startswith('@'), line

    tokens = _split_schedule_tokens(line[1:])  # strip '@', then split
    if not tokens:
        return None

    try:
        step_pct = float(tokens[0])
    except ValueError:
        return None

    ts_lo = ts_hi = None
    mode = ""
    lr_override = None
    dist_type = None
    dist_params = None

    _KNOWN_MODES = {"texture", "fullres"}

    for token in tokens[1:]:
        token = token.strip()
        if not token:
            continue

        if '=' in token:
            k, _, v = token.partition('=')
            k = k.strip().lower()
            v = v.strip()

            if k == 'range':
                parts = v.split('-', 1)
                if len(parts) == 2:
                    try:
                        ts_lo, ts_hi = int(parts[0]), int(parts[1])
                    except ValueError:
                        pass

            elif k == 'mode':
                mode = v.lower()
                if mode == '-':
                    mode = ''

            elif k == 'lr':
                try:
                    lr_override = float(v)
                except ValueError:
                    pass

            elif k == 'ts_dist':
                # v is e.g. "flow_shift(shift=3.0)" or "uniform" or "beta(alpha=0.14,beta=1.0)"
                m = _re.match(r'^(\w+)(?:\((.+)\))?$', v, _re.DOTALL)
                if m:
                    dist_type = m.group(1).lower()
                    raw_p = (m.group(2) or '').strip()
                    # _sample_timesteps expects space- or comma-separated "key=value" pairs
                    dist_params = raw_p  # commas inside parens are fine; parser splits on '='

        else:
            # bare word
            if token.lower() in _KNOWN_MODES:
                mode = token.lower()

    return (step_pct, ts_lo, ts_hi, mode, lr_override, dist_type, dist_params)


def _parse_ts_schedule_text(text: str):
    """Parse a timestep schedule from inline text (stored directly in the config).

    Two syntaxes are supported and may be mixed freely.

    **Keyword syntax (preferred)**::

        # @<step_pct>, [key=value | bare_mode] ...
        #
        # step_pct  0.0–1.0 fraction of total training steps at which row activates
        # range     lo-hi window, e.g. range=50-450
        # ts_dist   distribution + params, e.g. ts_dist=flow_shift(shift=3.0)
        #                                   or  ts_dist=beta(alpha=0.14,beta=1.0)
        # mode      texture | fullres | -     (bare word or mode=...)
        # lr        learning-rate override, e.g. lr=1e-4
        #
        # Unspecified fields inherit global config values.
        #
        @0,    range=200-800, ts_dist=flow_shift(shift=3.0), fullres, lr=1e-4
        @0.4,  range=0-700,   ts_dist=beta(alpha=0.14,beta=1.0), mode=texture
        @0.7,  range=0-1000,  ts_dist=uniform, mode=texture

    **Positional syntax (legacy)**::

        # Columns: step_pct   t_min   t_max   [mode]   [lr]
        0.00   200   800   fullres   1e-4
        0.40     0   700   texture   5e-5
        0.70     0  1000   texture

    Returns a sorted list of 7-tuples
    ``(step_pct, ts_lo, ts_hi, mode, lr_override, dist_type, dist_params)``
    where *dist_type* / *dist_params* are *None* when not specified (inherit global).
    Returns *None* if *text* is empty / contains no valid entries.
    """
    entries = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue  # ignore legacy section headers gracefully

        if line.startswith("@"):
            entry = _parse_new_schedule_line(line)
            if entry is not None:
                entries.append(entry)
            continue

        # --- Legacy positional format ---
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
                None,   # dist_type  → inherit global
                None,   # dist_params → inherit global
            )
        except (ValueError, IndexError):
            continue
        entries.append(entry)

    if not entries:
        return None
    entries.sort(key=lambda e: e[0])
    return entries   # list of (step_pct, ts_lo, ts_hi, mode, lr_override, dist_type, dist_params)


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


def text2cond(t, prompt):
    """Encode a text prompt into Anima conditioning tensors."""
    cond, _ = t.text_model.encode_text(prompt if isinstance(prompt, list) else [prompt])
    return cond, None


# --------------------------------------------------------------------------- #
# Debug / logging                                                              #
# --------------------------------------------------------------------------- #

CSVHEADS = ["network_rank", "network_alpha", "train_learning_rate", "train_iterations",
            "train_lr_scheduler", "model_version", "train_optimizer", "save_lora_name"]


def _clone_cond(cond):
    """Clone a conditioning tuple/tensor to CPU so it can be stored as a val snapshot."""
    if isinstance(cond, str):
        return cond
    if isinstance(cond, (tuple, list)):
        return tuple(c.detach().cpu().clone() if isinstance(c, torch.Tensor) else c for c in cond)
    if isinstance(cond, torch.Tensor):
        return cond.detach().cpu().clone()
    return cond


def savecsv(t, step, loss, lr, csvpath, copy=False, extra_cols=None):
    """Append one row to a training CSV log.

    extra_cols: optional dict {column_name: value} appended after the LR columns.
                Written into the header the first time the file is created.
    """
    header = []
    for key in CSVHEADS:
        header.append([key, getattr(t, key, "")])
    extra_names = list(extra_cols.keys()) if extra_cols else []
    header.append(["Step", "Loss"] + ["Learning Rate " + str(i + 1) for i in range(len(lr))] + extra_names)

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
        extra_vals = list(extra_cols.values()) if extra_cols else []
        writer.writerow([step, loss] + lr + extra_vals)


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
