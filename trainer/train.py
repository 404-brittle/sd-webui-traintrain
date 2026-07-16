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
from pprint import pprint
from typing import Optional
from accelerate.utils import set_seed
from math import sqrt as _sqrt

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
    from trainer import _anima_utils as anima_utils
    from trainer import qwen_image_autoencoder_kl

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

    _train_hybrid = getattr(t, 'train_hybrid_mode', False)  # legacy, kept for back-compat

    # VAE must stay alive for JIT encoding in texture mode.
    if not getattr(t, 'texture_mode', False):
        del t.vae
        if "BASE" not in t.network_blocks:
            del t.text_model

    flush()

    # Parse timestep curriculum schedule from inline config text (optional)
    _ts_schedule = _parse_ts_schedule_text(getattr(t, 'train_ts_schedule', '') or '')
    if _ts_schedule:
        print(f"Timestep schedule loaded: {len(_ts_schedule)} entries")

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                # Cast latents to train_model_precision (e.g. bf16) to match the model dtype.
                # Using train_lora_precision (default fp32) causes a dtype mismatch inside
                # Block._forward where torch.autocast(enabled=False) disables autocast,
                # so F.linear(float32_input, bf16_weight) fails without autocast to cast
                # the input. sd-scripts avoids this because its VAE caches latents in bf16.
                latents = batch["latent"].to(CUDA, dtype=t.train_model_precision)
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

                # Texture mode drives the active flow-shift value from the
                # epoch-based min/max schedule. Outside texture mode, the shift is
                # the static value parsed from `train_ts_dist_params`.
                if getattr(t, 'texture_mode', False):
                    epoch_now = getattr(getattr(t, 'dataloader', None), 'epoch', 0)
                    active_shift = _current_texture_shift(t, epoch_now)
                else:
                    active_shift = _parse_flow_shift(getattr(t, 'train_ts_dist_params', '') or '')

                dist_type  = getattr(t, 'train_timestep_distribution', 'flow_shift') or 'flow_shift'
                dist_params = getattr(t, 'train_ts_dist_params', '') or ''
                n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
                timesteps = _sample_timesteps(ts_lo, ts_hi, n_ts, CUDA, dist_type, active_shift, dist_params)
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

                # ── Monte-Carlo smoothed target (paper Section 3.1, Eq. 9) ──
                mc_smoothing = getattr(t, 'score_smoothing_mc', False)
                if mc_smoothing:
                    mc_samples = max(1, int(getattr(t, 'score_smoothing_mc_samples', 4) or 4))
                    mc_kappa = getattr(t, 'score_smoothing_kappa', 1.44) or 1.44
                    velocity_target = _mc_smooth_target(
                        latents, noise, timesteps,
                        n_samples=mc_samples, kappa=mc_kappa,
                    )

                train_mask = batch.get("mask")
                if train_mask is not None:
                    train_mask = train_mask.to(CUDA)

                loss, loss_ema, loss_velocity = process_loss(
                    t, model_pred, velocity_target, timesteps, loss_ema, loss_velocity,
                    mask=train_mask,
                )

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                _tile_now = _current_texture_tile_px(t, t.dataloader.epoch) if getattr(t, 'texture_mode', False) else None
                _shift_now = active_shift if getattr(t, 'texture_mode', False) else None
                _tex_tag = ""
                if _tile_now is not None:
                    _crop_aspect_str = ""
                    _aspect_text = getattr(t, 'texture_crop_aspect', '') or ''
                    if _aspect_text:
                        from trainer.dataset import _parse_aspect_ratios
                        _ar = _parse_aspect_ratios(_aspect_text)
                        if _ar:
                            # Pick the first matching-orientation ratio for display.
                            _is_wide = _tile_now >= _tile_now  # always True (square ref), so pick horizontal
                            _disp = [(w, h) for w, h in _ar if (w > h)]
                            if not _disp:
                                _disp = _ar
                            _w, _h = _disp[0]
                            _dir = "H" if _w > _h else "V"
                            _crop_aspect_str = f" Crop:{_w}:{_h}{_dir}"
                    _tex_tag = f" Tile: {_tile_now}px Shift: {_shift_now:.2f}{_crop_aspect_str}"
                pbar.set_description(
                    f"Loss EMA * 1000: {loss_ema * 1000:.4f}, LR: " + ", ".join(c_lrs) +
                    f", TS: {ts_lo}-{ts_hi}{_tex_tag}, Epoch: {t.dataloader.epoch}"
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
            active_shift = (_current_texture_shift(t, epoch)
                            if getattr(t, 'texture_mode', False)
                            else _parse_flow_shift(dist_params))
            n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
            band_lo = int(min(time_min, ts_range_hi - 1))
            band_hi = int(max(time_max, ts_range_lo + 1))
            timesteps = _sample_timesteps(band_lo, band_hi, n_ts, CUDA, dist_type, active_shift, dist_params)
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

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def _parse_flow_shift(dist_params: str) -> float:
    """Extract `shift=...` from the dist_params text; default 3.0."""
    for item in (dist_params or "").replace(",", " ").split():
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip() == "shift":
                try:
                    return float(v.strip())
                except ValueError:
                    pass
    return 3.0


def _parse_dist_params(dist_params: str) -> dict:
    params: dict = {}
    for item in (dist_params or "").replace(",", " ").split():
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                params[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return params


def _current_texture_tile_px(t, epoch: int) -> int:
    """Mirror of dataset._current_texture_tile_px — used by the progress bar."""
    min_tile = max(8, int(getattr(t, 'texture_min_tile', 256)))
    max_tile = max(min_tile, int(getattr(t, 'texture_max_tile', 1024)))
    snap = max(8, int(getattr(t, 'texture_tile_snap', 128)))
    step_epochs = max(1, int(getattr(t, 'texture_tile_step_epochs', 5)))
    stages = (max_tile - min_tile) // snap
    stage_index = min(stages, max(0, epoch // step_epochs))
    return max(min_tile, min(max_tile, min_tile + snap * stage_index))


def _current_texture_shift(t, epoch: int) -> float:
    """Active flow-shift for this epoch under the texture min/max schedule."""
    min_shift = float(getattr(t, 'texture_min_shift', 0.5))
    max_shift = float(getattr(t, 'texture_max_shift', 3.0))
    if max_shift < min_shift:
        min_shift, max_shift = max_shift, min_shift
    snap = max(8, int(getattr(t, 'texture_tile_snap', 128)))
    min_tile = max(8, int(getattr(t, 'texture_min_tile', 256)))
    max_tile = max(min_tile, int(getattr(t, 'texture_max_tile', 1024)))
    stages = max(1, (max_tile - min_tile) // snap)
    step_epochs = max(1, int(getattr(t, 'texture_shift_step_epochs', 5)))
    stage_index = min(stages, max(0, epoch // step_epochs))
    return min_shift + (max_shift - min_shift) * (stage_index / stages)


def _current_texture_crop_size(t, epoch: int, tile_px: int,
                                img_w: int, img_h: int):
    """Mirror of dataset._current_texture_crop_size — used by the progress bar."""
    # Import and delegate to the canonical implementation.
    from trainer.dataset import _current_texture_crop_size as _impl
    return _impl(t, epoch, tile_px, img_w, img_h)


def _sample_timesteps(ts_lo: int, ts_hi: int, n: int, device,
                       dist_type: str = "flow_shift", shift: float = 3.0,
                       dist_params: str = "") -> torch.Tensor:
    """Sample n integer timesteps in [ts_lo, ts_hi) using the specified distribution.

    dist_type:
      "uniform"      — flat torch.randint
      "flow_shift"   — bias toward high-noise via shift factor (sd-scripts convention).
                       The `shift` argument is used directly when > 1.0.
      "logit_normal" — sigmoid of Normal(mean, std); parameters read from dist_params
      "cosmap"       — cosine bijection; bias toward mid-noise
      "beta"         — Beta(alpha, beta); parameters read from dist_params
    """
    import math as _math

    span = ts_hi - ts_lo
    params = _parse_dist_params(dist_params)

    if dist_type == "uniform":
        return torch.randint(ts_lo, ts_hi, (n,), device=device)

    if dist_type == "logit_normal":
        mean = params.get("mean", 0.0)
        std = max(0.01, params.get("std", 1.0))
        u = torch.randn(n, device=device) * std + mean
        sigma = torch.sigmoid(u)
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    if dist_type == "cosmap":
        u = torch.rand(n, device=device).clamp(1e-6, 1.0 - 1e-6)
        sigma = 1.0 - 1.0 / (torch.tan(u * (_math.pi / 2.0)) + 1.0)
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    if dist_type == "beta":
        alpha = max(0.01, params.get("alpha", 0.5))
        beta_p = max(0.01, params.get("beta", 0.5))
        sigma = torch.distributions.Beta(
            torch.tensor(alpha, dtype=torch.float32, device=device),
            torch.tensor(beta_p, dtype=torch.float32, device=device),
        ).sample((n,))
        return (sigma * span + ts_lo).long().clamp(ts_lo, ts_hi - 1)

    # Default: flow_shift with the supplied shift value.
    shift = max(0.0, float(shift))
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
    # Cast to train_model_precision (e.g. bf16) so LoRA weights match the model dtype.
    # Using train_lora_precision (default fp32) causes a dtype mismatch when the model
    # runs under bf16 autocast: LoRA-patched layers return fp32 while adaln_lora_B_T_3D
    # from the timestep embedder is bf16, leading to RuntimeError at the + operator.
    return LoRANetwork(t).to(CUDA, dtype=t.train_model_precision)


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


def _score_smoothing_penalty(model_pred: torch.Tensor, timesteps: torch.Tensor,
                              kappa: float = 1.44) -> torch.Tensor:
    """Score-smoothing penalty inspired by the non-smoothness measure R[f] = ∫|f''(x)|dx.

    The paper proves that regularizing R[f] causes the NN to learn a smoothed
    version of the empirical score/velocity function, which drives interpolation
    rather than memorization. In the high-dimensional latent space of a DiT,
    we approximate this via a spatial Laplacian penalty on the predicted velocity
    field, weighted by a timestep-dependent factor δ(t) ∝ κ√t (Proposition 1).

    Args:
        model_pred: [B, C, H, W] predicted velocity.
        timesteps: [B] integer timesteps in [0, 1000].
        kappa: Smoothing strength parameter (δ = κ√(t/1000)).

    Returns:
        Scalar penalty value.
    """
    B, C, H, W = model_pred.shape
    # Normalised timesteps in [0, 1]
    t_norm = timesteps.float() / 1000.0  # [B]
    # δ(t) = κ√t — the smoothing window width (paper Proposition 1, Lemma 2)
    # The penalty is strongest at small t (where δ is small → less smoothing → more need to penalise)
    # We weight: w_smooth = 1 / (δ(t) + ε)  so small t gets high weight
    delta = kappa * torch.sqrt(t_norm.clamp(min=1e-6))  # [B]
    w_smooth = 1.0 / (delta + 0.01)  # [B], stronger penalty at small t

    # Spatial Laplacian penalty: ||Δ(velocity)||²  approximates ∫|f''(x)|dx
    # in the spatial dimensions of the latent space.
    # Finite-difference Laplacian: Δv ≈ v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j]
    laplacian = (
        F.pad(model_pred[:, :, :-1, :], (0, 0, 1, 0)) +
        F.pad(model_pred[:, :, 1:, :], (0, 0, 0, 1)) +
        F.pad(model_pred[:, :, :, :-1], (1, 0, 0, 0)) +
        F.pad(model_pred[:, :, :, 1:], (0, 1, 0, 0)) -
        4.0 * model_pred
    )
    # Mean squared Laplacian per sample → [B]
    penalty_per_sample = laplacian.pow(2).mean(dim=[1, 2, 3])

    # Weight by timestep-dependent factor and average
    weighted_penalty = (penalty_per_sample * w_smooth).mean()
    return weighted_penalty


def _mc_smooth_target(latents: torch.Tensor, noise: torch.Tensor,
                       timesteps: torch.Tensor, n_samples: int = 4,
                       kappa: float = 1.44) -> torch.Tensor:
    """Monte-Carlo smoothed velocity target — mirrors esf_mc_smoothed() from the paper.

    The paper (Section 3.1, Eq. 9) shows that averaging the ESF over a local
    window of width δ = κ√t produces a smoothed score that drives interpolation.
    Here we apply the same principle to the flow-matching velocity target by
    jittering the latents within a δ-sized window and averaging the resulting
    velocity targets.

    Args:
        latents: [B, C, H, W] clean latents.
        noise: [B, C, H, W] noise tensor.
        timesteps: [B] integer timesteps in [0, 1000].
        n_samples: Number of MC samples.
        kappa: Smoothing strength.

    Returns:
        [B, C, H, W] smoothed velocity target.
    """
    B, C, H, W = latents.shape
    t_norm = timesteps.float() / 1000.0  # [B]
    # δ(t) = κ√t — the smoothing window (paper Lemma 2: δ_t = κ√t)
    delta = kappa * torch.sqrt(t_norm.clamp(min=1e-6))  # [B]
    # Scale window to latent-space units (typical latent dim ~64-128, so
    # a δ of 0.05–0.2 corresponds to 3–13 pixels of jitter at 64px latents)
    delta = delta * (H / 64.0)  # scale by relative latent resolution

    # Generate jittered copies: [n_samples, B, C, H, W]
    jitter = torch.randn(n_samples, B, C, H, W, device=latents.device, dtype=latents.dtype)
    # Scale jitter by δ per batch element
    delta_4d = delta.view(B, 1, 1, 1)  # [B, 1, 1, 1]
    jitter = jitter * delta_4d.unsqueeze(0)  # [n_samples, B, C, H, W]

    # Compute velocity targets for each jittered copy
    # velocity = noise - (latents + jitter)
    velocities = []
    for k in range(n_samples):
        v = noise - (latents + jitter[k])
        velocities.append(v)

    smoothed = torch.stack(velocities, dim=0).mean(dim=0)
    return smoothed


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

    # ── Score smoothing penalty (paper Section 3.1, Proposition 1) ──────────
    smoothing_penalty_weight = getattr(t, 'score_smoothing_penalty', 0.0) or 0.0
    if smoothing_penalty_weight > 0.0:
        kappa = getattr(t, 'score_smoothing_kappa', 1.44) or 1.44
        penalty = _score_smoothing_penalty(original, timesteps, kappa=kappa)
        loss = loss + smoothing_penalty_weight * penalty

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

