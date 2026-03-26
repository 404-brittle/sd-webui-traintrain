#!/usr/bin/env python3
"""Automated LoRA layer scale optimisation using training data.

Given a trained LoRA and its original image-caption training pairs, learns a
per-layer scale factor α_i for each LoRA layer by minimising the denoising
loss on a held-out validation split of the training data.

Algorithm (Strategy 2A + ablation cut pass)
-------------------------------------------
  1. Encode all training images to latents (cached on CPU).
  2. Encode all captions to conditioning tensors (cached on CPU).
  3. Split into training and validation folds (default 85 / 15 %).
  4. Install forward hooks on all frozen DiT linear layers that inject:
         output += α_i · (lora_up @ lora_down @ x) · (arch_alpha / rank)
     where α_i is a learnable scalar nn.Parameter initialised to 1.0.
  5. Optimise {α_i} with Adam to minimise denoising loss on the training fold
     while tracking validation loss for early-stop / best-checkpoint selection.
  6. Ablation cut pass: for each layer (sorted by learned scale, lowest first),
     temporarily zero its scale and re-measure val loss.  If zeroing improves
     (or ties) val loss, the cut is kept and the new (lower) loss becomes the
     baseline for subsequent tests — so cuts compound naturally.  Otherwise the
     learned scale is restored.  This replaces fixed threshold-based cuts with a
     data-driven decision grounded in actual validation performance.
  7. Bake: multiply each layer's lora_up by its final scale.  Layers selected
     for cutting in step 6 are explicitly zeroed.  Save as a new .safetensors.

Usage
-----
    python tools/tune_lora_scales.py \\
        --lora_file   my_lora.safetensors \\
        --data_dir    /path/to/training_images \\
        --dit_path    /path/to/anima.safetensors \\
        --vae_path    /path/to/vae.safetensors \\
        --qwen3_path  /path/to/qwen3 \\
        --output      my_lora_tuned.safetensors

Notes
-----
  - Caption files must be .txt files sharing the stem of their paired image,
    e.g. image_001.png + image_001.txt.
  - Images are centre-cropped and resized to --image_size before encoding.
  - All latents/conditionings are cached once; subsequent steps are GPU-only.
  - The tool prints a per-layer scale table and saves a scale CSV alongside the
    output LoRA.
  - Typical settings: --steps 50 --lr 0.05 --batch_size 4
  - To allow aggressive cuts, lower --reg_strength or set it to 0.
  - To skip all cuts (only rescale), set --ablation_samples 0.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOL_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_SD_SCRIPTS_ROOT = os.environ.get("SD_SCRIPTS_PATH") or os.path.dirname(_ROOT_DIR)
if _SD_SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SD_SCRIPTS_ROOT)

from trainer.anima_support import (
    AnimaFlowScheduler,
    AnimaTextModel,
    anima_forward,
    move_cond_to_device,
)
from trainer.lora import (
    ANIMA_TARGET_REPLACE_MODULE,
    LORA_LINEAR,
    LORA_PREFIX_UNET,
    _matches_module_filter,
)

CUDA = torch.device("cuda:0")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Minimal forward context (matches pattern in other tools)
# ---------------------------------------------------------------------------

class _FwdCtx:
    def __init__(self, unet):
        self.unet = unet
        self.text_model = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _centre_crop_resize(img, size: int):
    """PIL image → square centre-crop then resize to (size, size)."""
    w, h = img.size
    s    = min(w, h)
    img  = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size))


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_data_dir(data_dir: str) -> list[tuple[str, str]]:
    """Walk *data_dir* for supported images and match them to .txt captions.

    Returns a list of (image_path, caption_text) pairs.
    Images without a matching .txt are skipped with a warning.
    """
    pairs = []
    for img_path in sorted(Path(data_dir).rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        cap_path = img_path.with_suffix(".txt")
        if not cap_path.exists():
            print(f"  Warning: no caption for {img_path.name} — skipped")
            continue
        caption = cap_path.read_text(encoding="utf-8").strip()
        if caption:
            pairs.append((str(img_path), caption))
    return pairs


# ---------------------------------------------------------------------------
# Latent + conditioning cache
# ---------------------------------------------------------------------------

def encode_dataset(
    pairs:      list[tuple[str, str]],
    vae,
    text_model: AnimaTextModel,
    image_size: int,
    dtype:      torch.dtype,
) -> list[dict]:
    """Encode every (image, caption) pair once and cache on CPU.

    Returns list of {"latent": Tensor[C,H,W], "cond": tuple} dicts.
    All tensors are moved to CPU after encoding to free VRAM.
    """
    import numpy
    from PIL import Image as PilImage

    records = []
    n = len(pairs)
    print(f"\nEncoding {n} training pairs ...")

    for i, (img_path, caption) in enumerate(pairs):
        # Image → latent
        with PilImage.open(img_path) as img:
            img = img.convert("RGB")
        img   = _centre_crop_resize(img, image_size)
        arr   = numpy.array(img).astype("float32") / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(CUDA, dtype=dtype)

        with torch.no_grad():
            latent = vae.encode_pixels_to_latents(img_t)  # [1, C, H, W]
        latent = latent.squeeze(0).cpu()  # [C, H, W]

        # Caption → conditioning (cached on CPU)
        cond, _ = text_model.encode_text([caption])
        cond = tuple(
            c.cpu() if isinstance(c, torch.Tensor) else c
            for c in cond
        )

        records.append({"latent": latent, "cond": cond})

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i + 1}/{n}]")

    torch.cuda.empty_cache()
    print(f"  Done — {len(records)} pairs cached on CPU.")
    return records


# ---------------------------------------------------------------------------
# LoRA layer map
# ---------------------------------------------------------------------------

def _block_of(key: str) -> int | None:
    """Return the DiT block index from a LoRA key, or None for non-block layers.

    Expects keys of the form ``lora_unet_blocks_N_...`` where N is an integer.
    """
    import re
    m = re.search(r"_blocks_(\d+)_", key)
    return int(m.group(1)) if m else None


def build_lora_layer_map(dit, module_filter: str = "") -> dict[str, nn.Module]:
    """Return {lora_name: nn.Linear} for all LoRA-targetable layers in the DiT.

    Uses the same naming scheme as the trainer so that names match keys in
    the LoRA .safetensors file.  No filter is applied by default so that
    every layer present in the LoRA can be mapped regardless of how it was
    trained.
    """
    layer_map: dict = {}
    for name, module in dit.named_modules():
        if module.__class__.__name__ not in ANIMA_TARGET_REPLACE_MODULE:
            continue
        for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ not in LORA_LINEAR:
                continue
            lora_name = (LORA_PREFIX_UNET + "." + name + "." + child_name).replace(".", "_")
            if module_filter and not _matches_module_filter(lora_name, module_filter):
                continue
            layer_map[lora_name] = child_module
    return layer_map


# ---------------------------------------------------------------------------
# Per-layer LoRA scale hooks
# ---------------------------------------------------------------------------

class LoRAScaleHooks:
    """Manages learnable scale parameters and forward hooks for LoRA layers.

    For each LoRA layer whose name appears in both the LoRA file and the DiT
    layer map, a forward hook is installed on the corresponding nn.Linear:

        output += α_i · (lora_up @ lora_down @ x) · (arch_alpha / rank)

    where α_i is a scalar nn.Parameter initialised to 1.0.

    When *block_level* is True, all layers that belong to the same DiT block
    (``blocks_N``) share a single nn.Parameter.  The optimiser then moves one
    scalar per block rather than one per layer, and the ablation pass tests
    entire blocks at once.

    Layers that exist in the LoRA file but are not found in the layer map
    (e.g. text-encoder layers) are tracked separately and baked at scale=1.0.
    """

    def __init__(
        self,
        lora_weights: dict[str, torch.Tensor],
        layer_map:    dict[str, nn.Module],
        block_level:  bool = False,
    ) -> None:
        base_keys = {
            k[: -len(".lora_down.weight")]
            for k in lora_weights
            if k.endswith(".lora_down.weight")
        }

        self.scales:      dict[str, nn.Parameter] = {}
        self.lora_data:   dict[str, tuple]         = {}  # base → (up, down, alpha, rank)
        self.unmapped:    set[str]                 = set()
        self._hooks:      list                     = []
        self.block_level: bool                     = block_level

        # In block_level mode, one shared Parameter per block index.
        # Non-block layers (block == None) each get their own Parameter.
        _block_params: dict[int, nn.Parameter] = {}

        for base in base_keys:
            up_key    = f"{base}.lora_up.weight"
            down_key  = f"{base}.lora_down.weight"
            alpha_key = f"{base}.alpha"

            lora_up   = lora_weights[up_key].float().to(CUDA)
            lora_down = lora_weights[down_key].float().to(CUDA)
            rank      = lora_down.shape[0]
            alpha_val = float(
                lora_weights[alpha_key].item() if alpha_key in lora_weights else rank
            )

            self.lora_data[base] = (lora_up, lora_down, alpha_val, rank)

            if base in layer_map:
                if block_level:
                    blk = _block_of(base)
                    if blk is not None:
                        if blk not in _block_params:
                            _block_params[blk] = nn.Parameter(torch.ones(1, device=CUDA))
                        self.scales[base] = _block_params[blk]
                    else:
                        # Non-block layer — own parameter
                        self.scales[base] = nn.Parameter(torch.ones(1, device=CUDA))
                else:
                    self.scales[base] = nn.Parameter(torch.ones(1, device=CUDA))
            else:
                self.unmapped.add(base)

        n_params = len(self.parameters())
        mode_str = f"block-level ({n_params} blocks)" if block_level else f"layer-level ({n_params} layers)"
        print(
            f"\nLoRA layers: {len(base_keys)} total, "
            f"{len(self.scales)} mapped to DiT [{mode_str}], "
            f"{len(self.unmapped)} unmapped (baked unchanged)"
        )

    # ------------------------------------------------------------------

    def install(self, layer_map: dict[str, nn.Module]) -> None:
        """Register forward hooks on all mapped DiT linear layers."""
        if self._hooks:
            raise RuntimeError("Hooks already installed — call remove() first.")
        for base, scale_param in self.scales.items():
            if base not in layer_map:
                continue
            lora_up, lora_down, alpha_val, rank = self.lora_data[base]
            hook_fn = _make_lora_hook(lora_up, lora_down, alpha_val, rank, scale_param)
            self._hooks.append(layer_map[base].register_forward_hook(hook_fn))

    def remove(self) -> None:
        """Detach all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def parameters(self) -> list[nn.Parameter]:
        """Return unique Parameter objects (block-level mode shares params across layers)."""
        seen, unique = set(), []
        for p in self.scales.values():
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    def learned_scales(self) -> dict[str, float]:
        return {k: v.item() for k, v in self.scales.items()}


def _make_lora_hook(
    lora_up:     torch.Tensor,
    lora_down:   torch.Tensor,
    alpha_val:   float,
    rank:        int,
    scale_param: nn.Parameter,
):
    """Return a forward hook that injects α_i · LoRA(x) into the layer output.

    The LoRA weight matrices and scale factor are treated as constants
    (computed under no_grad).  Only scale_param participates in the
    gradient graph, so backward() correctly computes ∂loss/∂α_i = Σ(∂loss/∂out · delta).
    """
    arch_scale = alpha_val / rank

    def hook(_module, inp, out):
        x = inp[0]
        # Delta is computed without tracking gradients for x or LoRA weights.
        # The multiplication by scale_param (which has requires_grad=True)
        # re-attaches to the computation graph at exactly one edge per layer.
        with torch.no_grad():
            delta = (x.float() @ lora_down.t() @ lora_up.t()) * arch_scale
            delta = delta.to(out.dtype)
        return out + scale_param * delta

    return hook


# ---------------------------------------------------------------------------
# Full denoising + preview helpers
# ---------------------------------------------------------------------------

def _build_timestep_schedule(n_steps: int, schedule: str) -> list[float]:
    """Return a list of n_steps+1 timestep fractions in [1.0 → 0.0].

    schedule='uniform'  — linear spacing (standard Euler).
    schedule='beta'     — Beta(0.6, 0.6) distribution, denser near the
                          middle of the trajectory where the model has most
                          influence; sparser near t=0 and t=1.  Falls back
                          to uniform if scipy is not installed.
    """
    if schedule == "beta":
        try:
            from scipy.stats import beta as _scipy_beta
            # ppf maps (0,1) → (0,1); we want t going 1→0 across n_steps+1 points
            ts = [
                float(1.0 - _scipy_beta.ppf(i / n_steps, 0.6, 0.6))
                for i in range(n_steps + 1)
            ]
            return ts
        except ImportError:
            print("[preview] scipy not found — falling back to uniform schedule")
    # uniform
    return [1.0 - i / n_steps for i in range(n_steps + 1)]


def denoise_latent(
    fwd_ctx:   _FwdCtx,
    latent_shape: tuple,
    cond,
    dtype:     torch.dtype,
    n_steps:   int = 20,
    sampler:   str = "euler",
    schedule:  str = "uniform",
    uncond=None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Denoise from pure noise to a clean latent using rectified flow matching.

    sampler:
      'euler'            — deterministic ODE step: x = x − v·dt
      'euler_ancestral'  — stochastic; at each step reconstructs x₀ and
                           re-noises to the next sigma level.  Produces more
                           diverse / natural textures than plain Euler.
                           Aliased as 'er_sde'.

    schedule:
      'uniform'  — linearly-spaced timesteps from t=1 → t=0.
      'beta'     — Beta(0.6,0.6)-distributed timesteps (denser near t≈0.5);
                   matches ComfyUI's beta schedule for FLUX-family models.
                   Requires scipy; falls back to uniform if unavailable.

    uncond / cfg_scale:
      When uncond is provided and cfg_scale > 1.0, classifier-free guidance
      is applied at every step:
          v = v_uncond + cfg_scale * (v_cond − v_uncond)
      uncond should be the encoded negative-prompt conditioning tuple,
      pre-moved to the correct device/dtype.

    Returns the denoised latent tensor [1, C, H, W] on CUDA.
    """
    x        = torch.randn(latent_shape, device=CUDA, dtype=dtype)
    ts       = _build_timestep_schedule(n_steps, schedule)
    ancestral = sampler in ("euler_ancestral", "er_sde")
    use_cfg  = (uncond is not None and cfg_scale > 1.0)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            for i in range(n_steps):
                t_cur  = ts[i]
                t_next = ts[i + 1]
                t_int  = torch.tensor(
                    [max(1, round(t_cur * 1000))], device=CUDA, dtype=torch.long
                )

                if use_cfg:
                    v_cond  = anima_forward(fwd_ctx, x, t_int, cond)
                    v_uncond = anima_forward(fwd_ctx, x, t_int, uncond)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = anima_forward(fwd_ctx, x, t_int, cond)

                if ancestral and t_next > 0.0:
                    # Reconstruct x0 from current noisy latent and velocity.
                    # Flow matching: x_t = (1−t)·x0 + t·noise  →  x0 = x_t − t·v
                    x0_pred = (x.float() - t_cur * v.float()).to(dtype)
                    # Re-noise to t_next level with fresh noise.
                    noise = torch.randn_like(x)
                    x = ((1.0 - t_next) * x0_pred + t_next * noise).to(dtype)
                else:
                    # Euler step
                    dt = t_cur - t_next
                    x  = (x - v.float() * dt).to(dtype)

    return x


def save_preview_image(latent: torch.Tensor, vae, path: str) -> None:
    """Decode a latent and save as a PNG preview image."""
    from PIL import Image as PilImage
    import numpy as np

    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent)           # [1, 3, H, W] in [−1, 1]

    arr = (
        pixels.squeeze(0).float().cpu().clamp(-1.0, 1.0)
        .add(1.0).mul(127.5).byte()
        .permute(1, 2, 0).numpy()
    )
    PilImage.fromarray(arr).save(path)


def make_preview_fn(dit, vae, val_data: list[dict], args, output_path: str, text_model=None):
    """Return a callable ``preview_fn(label)`` that generates and saves a
    fully denoised preview image.

    Prompt sourcing:
      - If --preview_prompt is set, it is encoded via text_model.
      - Otherwise the first validation record's cached conditioning is used.
    CFG:
      - If --preview_cfg_scale > 1.0, classifier-free guidance is applied.
        --preview_negative_prompt is encoded as the unconditional input
        (empty string is used when the flag is not set).

    The function is a no-op when *vae* is None or --preview_steps is 0.
    Files are named: ``<stem>_<unix_time>_<label>.png`` in the output dir.
    """
    if vae is None or args.preview_steps == 0:
        return None

    dtype        = _parse_dtype(args.precision)
    out_dir      = str(Path(output_path).parent)
    out_stem     = Path(output_path).stem
    latent_shape = (1, *val_data[0]["latent"].shape)
    cfg_scale    = getattr(args, "preview_cfg_scale", 1.0)

    # --- positive conditioning ---
    if args.preview_prompt and text_model is not None:
        cond_cpu, _ = text_model.encode_text([args.preview_prompt])
        cond_cpu = tuple(c.cpu() if isinstance(c, torch.Tensor) else c for c in cond_cpu)
        print(f"  [preview] positive prompt: {args.preview_prompt!r}")
    else:
        cond_cpu = val_data[0]["cond"]
        if not args.preview_prompt:
            print("  [preview] positive prompt: <first training caption>")

    # --- negative conditioning (for CFG) ---
    uncond_cpu = None
    if cfg_scale > 1.0 and text_model is not None:
        neg = getattr(args, "preview_negative_prompt", "") or ""
        uncond_raw, _ = text_model.encode_text([neg])
        uncond_cpu = tuple(c.cpu() if isinstance(c, torch.Tensor) else c for c in uncond_raw)
        print(f"  [preview] negative prompt: {neg!r}  cfg_scale={cfg_scale}")

    def _fn(label: str) -> None:
        cond   = move_cond_to_device(cond_cpu, CUDA, dtype)
        uncond = move_cond_to_device(uncond_cpu, CUDA, dtype) if uncond_cpu is not None else None
        fwd_ctx = _FwdCtx(dit)
        latent  = denoise_latent(
            fwd_ctx, latent_shape, cond, dtype,
            n_steps=args.preview_steps,
            sampler=args.preview_sampler,
            schedule=args.preview_schedule,
            uncond=uncond,
            cfg_scale=cfg_scale,
        )
        ts   = int(time.time())
        path = os.path.join(out_dir, f"{out_stem}_{ts}_{label}.png")
        save_preview_image(latent, vae, path)
        print(f"    [preview] {path}")

    return _fn


# ---------------------------------------------------------------------------
# Denoising loss
# ---------------------------------------------------------------------------

def compute_loss(
    fwd_ctx:   _FwdCtx,
    scheduler: AnimaFlowScheduler,
    records:   list[dict],
    n_samples: int,
    dtype:     torch.dtype,
    min_ts:    int,
    max_ts:    int,
) -> torch.Tensor:
    """Mean denoising loss over *n_samples* randomly drawn (latent, cond, t, ε) tuples.

    Gradient graph is preserved so that loss.backward() propagates through the
    LoRA scale parameters via the forward hooks.
    """
    total = torch.zeros(1, device=CUDA)

    for _ in range(n_samples):
        rec    = records[torch.randint(len(records), (1,)).item()]
        latent = rec["latent"].unsqueeze(0).to(CUDA, dtype=dtype)   # [1, C, H, W]
        cond   = move_cond_to_device(rec["cond"], CUDA, dtype)

        noise  = torch.randn_like(latent)
        t      = torch.randint(min_ts, max_ts, (1,), device=CUDA)
        noisy  = scheduler.add_noise(latent, noise, t)

        with torch.autocast("cuda", dtype=dtype):
            pred = anima_forward(fwd_ctx, noisy, t, cond)

        # Flow-matching velocity target: v = ε − x₀
        velocity_target = (noise - latent).to(torch.float32)
        total = total + F.mse_loss(pred.float(), velocity_target)

    return total / n_samples


# ---------------------------------------------------------------------------
# LPIPS perceptual loss (cut-pass alternative to MSE)
# ---------------------------------------------------------------------------

def _load_lpips(device=None):
    """Import and cache an LPIPS VGG network.  Requires: pip install lpips"""
    try:
        import lpips as _lpips_lib
    except ImportError:
        raise ImportError(
            "The 'lpips' package is required for --metric lpips.\n"
            "Install it with:  pip install lpips"
        )
    fn = _lpips_lib.LPIPS(net="vgg", verbose=False)
    fn.eval()
    if device is not None:
        fn = fn.to(device)
    return fn


def _load_dists(device=None):
    """Import and cache a DISTS network.  Requires: pip install piq

    DISTS (Deep Image Structure and Texture Similarity) explicitly models
    both structural and textural similarity, making it sensitive to gloss,
    specularity, fabric weave, brushstrokes and other fine-grained texture
    differences that LPIPS/VGG tends to miss.
    """
    try:
        import piq as _piq
    except ImportError:
        raise ImportError(
            "The 'piq' package is required for --metric dists.\n"
            "Install it with:  pip install piq"
        )
    fn = _piq.DISTS(reduction="mean")
    fn.eval()
    if device is not None:
        fn = fn.to(device)
    return fn


# ---------------------------------------------------------------------------
# Gram matrix style loss  (positionally invariant texture metric)
# ---------------------------------------------------------------------------

def _gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Normalised Gram matrix [B, C, C] from feature map [B, C, H, W].

    Collapses spatial dimensions entirely — only feature co-occurrence
    statistics (texture) survive.  Fully positionally invariant.
    """
    B, C, H, W = features.shape
    f = features.view(B, C, -1)                        # [B, C, HW]
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)  # [B, C, C]


def _load_gram(device=None):
    """Multi-scale VGG16 Gram matrix style loss.

    Extracts features at four ReLU stages and computes Gram MSE at each:
      relu1_2 (idx 3)  — grain, noise, fine glitch, sub-pixel texture
      relu2_2 (idx 8)  — blur kernel statistics, edge softness, halation
      relu3_3 (idx 15) — bokeh/DoF shape, mid-level texture, lens aberration
      relu4_3 (idx 22) — colour grading, tonal character, painterly style

    Fully positionally invariant: the Gram matrix discards all spatial
    information, retaining only feature co-occurrence statistics.
    Inputs expected in [−1, 1].  Uses torchvision (no extra install needed).
    """
    import torchvision.models as tvm

    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features.eval()
    vgg.requires_grad_(False)
    if device is not None:
        vgg = vgg.to(device)

    _TARGETS = {3: 1.0, 8: 1.0, 15: 1.0, 22: 1.0}   # layer_idx → weight
    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Normalise from [−1,1] to ImageNet statistics
        a_n = (a.float() * 0.5 + 0.5 - _mean) / _std
        b_n = (b.float() * 0.5 + 0.5 - _mean) / _std
        total = torch.zeros(1, device=a.device)
        xa, xb = a_n, b_n
        for i, layer in enumerate(vgg):
            xa = layer(xa)
            xb = layer(xb)
            if i in _TARGETS:
                total = total + _TARGETS[i] * F.mse_loss(
                    _gram_matrix(xa), _gram_matrix(xb)
                )
        return total

    return _fn


# ---------------------------------------------------------------------------
# FFT power-spectrum loss  (frequency-domain, perfectly shift-invariant)
# ---------------------------------------------------------------------------

def _fft_spectrum_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2-D FFT log-power-spectrum L2 distance.

    By the Fourier shift theorem |FFT(shifted image)| = |FFT(image)|, so
    this metric is perfectly shift-invariant.  It directly captures:
      • Blur          — attenuated high-frequency power
      • Film grain    — elevated broadband high-frequency power
      • Lens noise    — characteristic spectral elevation profile
      • Compression   — periodic ringing peaks in frequency space
      • Glitch        — spikes / asymmetries in the spectrum

    No network required.  Inputs in any pixel range (log1p normalises scale).
    """
    def _log_power(img: torch.Tensor) -> torch.Tensor:
        fft   = torch.fft.rfft2(img.float())          # [B, C, H, W//2+1] complex
        power = fft.real ** 2 + fft.imag ** 2         # magnitude squared
        return torch.log1p(power)                      # log for numerical range

    return F.mse_loss(_log_power(a), _log_power(b))


def _load_perceptual(metric: str, device=None):
    """Return a perceptual loss callable ``fn(img_a, img_b) → scalar Tensor``.

    Both inputs are expected in [−1, 1].  Metric-specific normalisation is
    handled internally so callers do not need to know about it.

    Supported metrics:
      'lpips'     — LPIPS VGG perceptual (requires: pip install lpips)
      'dists'     — DISTS structure + texture (requires: pip install piq)
      'gram'      — Multi-scale VGG Gram matrix; positionally invariant;
                    best for stylistic / adverse-image LoRAs
      'fft'       — 2-D FFT log-power spectrum; perfectly shift-invariant;
                    best for blur, grain, noise, glitch sensitivity
      'gram_fft'  — Gram + FFT combined; most comprehensive style metric
    """
    if metric == "lpips":
        return _load_lpips(device)

    elif metric == "dists":
        raw = _load_dists(device)
        def _dists_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a01 = (a.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            b01 = (b.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            return raw(a01, b01)
        return _dists_fn

    elif metric == "gram":
        return _load_gram(device)

    elif metric == "fft":
        return _fft_spectrum_fn

    elif metric == "gram_fft":
        gram_fn = _load_gram(device)
        def _gram_fft_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return gram_fn(a, b) + _fft_spectrum_fn(a, b)
        return _gram_fft_fn

    else:
        raise ValueError(f"Unknown perceptual metric: {metric!r}")


def compute_perceptual_loss(
    fwd_ctx:      _FwdCtx,
    scheduler:    AnimaFlowScheduler,
    records:      list[dict],
    n_samples:    int,
    dtype:        torch.dtype,
    min_ts:       int,
    max_ts:       int,
    vae,
    perceptual_fn,
) -> torch.Tensor:
    """Perceptual loss via single-step x₀ reconstruction + perceptual metric.

    For each sample:
      1. Corrupt the cached latent at a random timestep t.
      2. Run the DiT (with current LoRA hooks) to predict velocity v.
      3. Reconstruct clean latent: x₀_pred = noisy − (t/1000) · v_pred
      4. Decode both x₀_pred and the reference latent to pixels via VAE.
      5. Compute perceptual distance between predicted and reference pixels.

    Returns the mean distance as a Tensor so that callers in the
    optimisation loop can call .backward() through it (gradient flows:
    metric → VAE decode → x₀_pred → v_pred → LoRA scale params).
    Callers that don't need gradients should wrap in torch.no_grad().
    Pixels are decoded at full VAE resolution and downsampled to 512 px
    before the metric to bound VRAM usage.
    """
    total = torch.zeros(1, device=CUDA)

    for _ in range(n_samples):
        rec    = records[torch.randint(len(records), (1,)).item()]
        latent = rec["latent"].unsqueeze(0).to(CUDA, dtype=dtype)
        cond   = move_cond_to_device(rec["cond"], CUDA, dtype)

        noise = torch.randn_like(latent)
        t     = torch.randint(min_ts, max_ts, (1,), device=CUDA)
        noisy = scheduler.add_noise(latent, noise, t)

        with torch.autocast("cuda", dtype=dtype):
            v_pred = anima_forward(fwd_ctx, noisy, t, cond)

        # x₀ reconstruction: noisy = (1-σ)·x₀ + σ·noise  →  x₀ = noisy − σ·v
        sigma   = (t.float() / 1000.0).view(-1, 1, 1, 1)
        x0_pred = (noisy.float() - sigma * v_pred.float()).to(dtype)

        # Decode to pixels in [−1, 1]; reference decoded under no_grad
        pixels_pred = vae.decode_to_pixels(x0_pred)             # [1, 3, H, W]
        with torch.no_grad():
            pixels_ref = vae.decode_to_pixels(latent)           # [1, 3, H, W]

        # Resize to max 512 px for VRAM efficiency
        if pixels_pred.shape[-1] > 512 or pixels_pred.shape[-2] > 512:
            pixels_pred = F.interpolate(
                pixels_pred.float(), size=512, mode="bilinear", align_corners=False,
            )
            pixels_ref = F.interpolate(
                pixels_ref.float(), size=512, mode="bilinear", align_corners=False,
            )

        total = total + perceptual_fn(pixels_pred.float(), pixels_ref.float()).mean()

    return total / n_samples


# ---------------------------------------------------------------------------
# Optimisation loop
# ---------------------------------------------------------------------------

def run_optimization(
    dit,
    scheduler:  AnimaFlowScheduler,
    hooks:      LoRAScaleHooks,
    layer_map:  dict[str, nn.Module],
    train_data: list[dict],
    val_data:   list[dict],
    args,
    vae=None,
    perceptual_fn=None,
    preview_fn=None,
) -> tuple[dict[str, float], float]:
    """Optimise per-layer LoRA scale parameters via Adam.

    When args.metric is a perceptual metric (lpips/dists), both training and
    validation loss use compute_perceptual_loss (via single-step x₀ reconstruction).
    The gradient path is: metric → VAE decode → x₀_pred → v_pred → scale_param.
    Otherwise (args.metric == 'mse'), the standard denoising MSE is used.

    Returns (best_scales, best_val_loss) where best_scales is the
    {base_key: float} dict from the step with minimum validation loss.
    """
    dtype            = _parse_dtype(args.precision)
    use_perceptual   = (args.metric != "mse")

    hooks.install(layer_map)
    fwd_ctx = _FwdCtx(dit)

    params = hooks.parameters()
    optim  = torch.optim.Adam(params, lr=args.lr)

    # Resolve warmup: int >= 1 → absolute step count; float in (0,1) → fraction.
    _ws = getattr(args, "warmup_steps", 0.0)
    warmup_n = (
        max(1, round(_ws * args.steps)) if 0.0 < _ws < 1.0
        else int(_ws)
    )

    def _scheduled_lr(step: int) -> float:
        """Linear warmup for the first warmup_n steps, then constant."""
        if warmup_n > 0 and step <= warmup_n:
            return args.lr * step / warmup_n
        return args.lr

    best_val_loss  = float("inf")
    best_scales    = hooks.learned_scales()
    no_improve_cnt = 0

    warmup_str = (
        f"{warmup_n} steps ({_ws:.0%})" if 0.0 < _ws < 1.0
        else (f"{warmup_n} steps" if warmup_n > 0 else "disabled")
    )
    print(f"\n{'=' * 60}")
    print(f"Optimising {len(params)} scale parameters")
    print(f"  train / val     : {len(train_data)} / {len(val_data)} samples")
    print(f"  metric          : {args.metric.upper()}")
    print(f"  steps           : {args.steps}")
    print(f"  lr              : {args.lr}")
    print(f"  warmup          : {warmup_str}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  reg_strength    : {args.reg_strength}")
    print(f"  patience        : {args.patience if args.patience > 0 else 'disabled'}")
    print(f"  timestep range  : [{args.min_timestep}, {args.max_timestep}]")
    print(f"{'=' * 60}\n")

    def _val_loss() -> float:
        with torch.no_grad():
            if use_perceptual:
                return compute_perceptual_loss(
                    fwd_ctx, scheduler, val_data,
                    args.batch_size, dtype, args.min_timestep, args.max_timestep,
                    vae, perceptual_fn,
                ).item()
            return compute_loss(
                fwd_ctx, scheduler, val_data,
                args.batch_size, dtype, args.min_timestep, args.max_timestep,
            ).item()

    try:
        for step in range(1, args.steps + 1):
            # --- LR schedule (warmup) ---
            current_lr = _scheduled_lr(step)
            for pg in optim.param_groups:
                pg["lr"] = current_lr

            # --- training step ---
            optim.zero_grad()

            if use_perceptual:
                # Gradient accumulation: one sample at a time so only one forward
                # graph (DiT + VAE decode + perceptual net) lives in VRAM at once.
                # Each sample's graph is freed immediately after its .backward().
                # Dividing by batch_size before backward makes the summed gradients
                # equal to the mean gradient across the batch.
                train_loss_val = 0.0
                for _ in range(args.batch_size):
                    sl = compute_perceptual_loss(
                        fwd_ctx, scheduler, train_data,
                        1, dtype, args.min_timestep, args.max_timestep,
                        vae, perceptual_fn,
                    )
                    if args.reg_strength > 0.0:
                        stacked = torch.stack(params)
                        sl = sl + args.reg_strength * ((stacked - 1.0) ** 2).mean()
                    (sl / args.batch_size).backward()
                    train_loss_val += sl.detach().item()
                train_loss_val /= args.batch_size
                reg = torch.zeros(1, device=CUDA)  # already folded into sl above
            else:
                train_loss = compute_loss(
                    fwd_ctx, scheduler, train_data,
                    args.batch_size, dtype, args.min_timestep, args.max_timestep,
                )
                if args.reg_strength > 0.0:
                    stacked = torch.stack(params)
                    reg = args.reg_strength * ((stacked - 1.0) ** 2).mean()
                    loss = train_loss + reg
                else:
                    reg  = torch.zeros(1, device=CUDA)
                    loss = train_loss
                loss.backward()
                train_loss_val = train_loss.item()

            optim.step()

            # --- validation (no gradient tracking needed) ---
            val_loss = _val_loss()

            # Track best checkpoint by val loss
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_scales    = hooks.learned_scales()
                no_improve_cnt = 0
                marker = "*"
                if preview_fn:
                    preview_fn(f"opt_step{step:04d}")
            else:
                no_improve_cnt += 1
                marker = " "

            print(
                f"  [{step:>4}/{args.steps}]{marker} "
                f"train={train_loss_val:.5f}  "
                f"val={val_loss:.5f}  "
                f"reg={reg.item():.5f}  "
                f"lr={current_lr:.2e}"
            )

            if args.patience > 0 and no_improve_cnt >= args.patience:
                print(f"\n  Early stop: val loss stagnant for {args.patience} steps.")
                break

    except KeyboardInterrupt:
        print("\n\n  [Interrupted] Returning best scales found so far.")

    finally:
        hooks.remove()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  Best val loss: {best_val_loss:.5f}  (at step with '*' above)")
    return best_scales, best_val_loss


# ---------------------------------------------------------------------------
# Ablation cut pass
# ---------------------------------------------------------------------------

def ablation_cut_pass(
    dit,
    scheduler:    AnimaFlowScheduler,
    hooks:        LoRAScaleHooks,
    layer_map:    dict[str, nn.Module],
    val_data:     list[dict],
    best_scales:  dict[str, float],
    baseline_val_loss: float,
    args,
    vae=None,
    perceptual_fn=None,
    preview_fn=None,
) -> dict[str, float]:
    """Data-driven layer cutting via sequential val-loss ablation.

    For each LoRA layer (sorted ascending by learned scale — lowest-scale layers
    are most likely to be cut and are tested first), the layer's scale is
    temporarily set to zero while all other layers retain their current values.
    The metric (MSE, LPIPS, or DISTS) is measured with --ablation_samples samples.

    Cut decision:
        zeroed_loss  <=  current_baseline  →  keep cut, update baseline
        zeroed_loss  >   current_baseline  →  restore scale

    Cuts compound: once a layer is cut, it stays at zero for all subsequent
    tests, so the baseline degrades correctly as layers are removed.

    Returns the final {base_key: float} scale dict with cut layers set to 0.0.
    """
    dtype          = _parse_dtype(args.precision)
    use_perceptual = (args.metric != "mse")

    # Load best scales into the Parameter objects and install hooks
    for base, scale_val in best_scales.items():
        if base in hooks.scales:
            hooks.scales[base].data.fill_(scale_val)
    hooks.install(layer_map)

    fwd_ctx      = _FwdCtx(dit)
    final_scales = dict(best_scales)
    baseline     = baseline_val_loss
    cut_count    = 0

    # ------------------------------------------------------------------
    # Build candidate list: either per-layer or per-block.
    # In block mode, layers sharing the same Parameter are grouped; the
    # "scale" used for sorting is the mean absolute value across the group.
    # ------------------------------------------------------------------
    if args.block_level:
        from collections import defaultdict
        block_groups: dict[int | str, list[tuple[str, float]]] = defaultdict(list)
        for base, scale_val in best_scales.items():
            if base not in hooks.scales:
                continue
            blk = _block_of(base)
            key = blk if blk is not None else f"non_block_{base}"
            block_groups[key].append((base, scale_val))

        # Sort blocks by mean |scale| ascending — weakest blocks tested first
        block_candidates = sorted(
            block_groups.items(),
            key=lambda kv: sum(abs(sv) for _, sv in kv[1]) / len(kv[1]),
        )
        n = len(block_candidates)
        unit_label = "blocks"
    else:
        layer_candidates = [
            (base, scale_val)
            for base, scale_val in best_scales.items()
            if base in hooks.scales
        ]
        layer_candidates.sort(key=lambda x: abs(x[1]))
        n = len(layer_candidates)
        unit_label = "layers"

    metric_label = args.metric.upper() if use_perceptual else "MSE loss"
    print(f"\n{'=' * 60}")
    print(f"Ablation cut pass — {n} {unit_label} to test")
    print(f"  Baseline {metric_label:<12}: {baseline:.5f}")
    print(f"  Metric            : {metric_label}")
    print(f"  Granularity       : {'block' if args.block_level else 'layer'}")
    print(f"  Ablation samples  : {args.ablation_samples}")
    print(f"{'=' * 60}\n")

    def _measure() -> float:
        with torch.no_grad():
            if use_perceptual:
                return compute_perceptual_loss(
                    fwd_ctx, scheduler, val_data,
                    args.ablation_samples, dtype,
                    args.min_timestep, args.max_timestep,
                    vae, perceptual_fn,
                ).item()
            return compute_loss(
                fwd_ctx, scheduler, val_data,
                args.ablation_samples, dtype,
                args.min_timestep, args.max_timestep,
            ).item()

    try:
        if args.block_level:
            for i, (blk_key, layers) in enumerate(block_candidates):
                mean_scale = sum(abs(sv) for _, sv in layers) / len(layers)
                # Zero all layers in this block via shared param
                hooks.scales[layers[0][0]].data.fill_(0.0)

                zeroed_loss = _measure()

                if zeroed_loss <= baseline:
                    for base, _ in layers:
                        final_scales[base] = 0.0
                    old_baseline = baseline
                    baseline  = zeroed_loss
                    cut_count += 1
                    verdict = f"CUT   (loss {zeroed_loss:.5f}, was {old_baseline:.5f})"
                    if preview_fn:
                        lbl = f"abl_block{blk_key:02d}" if isinstance(blk_key, int) else f"abl_{blk_key}"
                        preview_fn(lbl)
                else:
                    # Restore — all layers share the same param, restore once
                    hooks.scales[layers[0][0]].data.fill_(mean_scale)
                    verdict = f"keep  (would raise loss {zeroed_loss:.5f} vs {baseline:.5f})"

                blk_label = f"block {blk_key}" if isinstance(blk_key, int) else str(blk_key)
                print(f"  [{i + 1:>3}/{n}] ᾱ={mean_scale:+.4f}  {blk_label:<20} "
                      f"({len(layers)} layers)  {verdict}")
        else:
            for i, (base, scale_val) in enumerate(layer_candidates):
                hooks.scales[base].data.fill_(0.0)

                zeroed_loss = _measure()

                if zeroed_loss <= baseline:
                    final_scales[base] = 0.0
                    old_baseline = baseline
                    baseline  = zeroed_loss
                    cut_count += 1
                    verdict = f"CUT   (loss {zeroed_loss:.5f}, was {old_baseline:.5f})"
                    if preview_fn:
                        short_lbl = "_".join(base.split("_")[-4:])
                        preview_fn(f"abl_{short_lbl}")
                else:
                    hooks.scales[base].data.fill_(scale_val)
                    verdict = f"keep  (would raise loss {zeroed_loss:.5f} vs {baseline:.5f})"

                short = "_".join(base.split("_")[-5:])
                print(f"  [{i + 1:>4}/{n}] α={scale_val:+.4f}  {short:<40}  {verdict}")

    except KeyboardInterrupt:
        print(f"\n\n  [Interrupted] Returning cuts decided so far ({cut_count}/{n}).")

    finally:
        hooks.remove()
        torch.cuda.empty_cache()

    print(f"\n  Ablation complete — {cut_count}/{n} {unit_label} cut")
    print(f"  Final baseline val loss: {baseline:.5f}  "
          f"(Δ {baseline - baseline_val_loss:+.5f} vs post-optimisation)")
    return final_scales


# ---------------------------------------------------------------------------
# Scale visualisation
# ---------------------------------------------------------------------------

# Canonical column grouping for the Anima DiT block structure.
# Each group is (display_label, [sublayer_name, ...]).
_COL_GROUPS: list[tuple[str, list[str]]] = [
    ("Self-Attn", [
        "self_attn_q_proj", "self_attn_k_proj",
        "self_attn_v_proj", "self_attn_output_proj",
    ]),
    ("Cross-Attn", [
        "cross_attn_q_proj", "cross_attn_k_proj",
        "cross_attn_v_proj", "cross_attn_output_proj",
    ]),
    ("MLP", [
        "mlp_layer1", "mlp_layer2",
    ]),
    ("AdaLN", [
        "adaln_modulation_self_attn_1", "adaln_modulation_self_attn_2",
        "adaln_modulation_cross_attn_1", "adaln_modulation_cross_attn_2",
        "adaln_modulation_mlp_1",        "adaln_modulation_mlp_2",
    ]),
]
_ALL_SUBLAYERS: list[str] = [s for _, subs in _COL_GROUPS for s in subs]
_N_BLOCKS = 28


def plot_scales(scales: dict[str, float], output_path: str) -> None:
    """Save a heatmap visualisation of per-layer LoRA scales as a PNG.

    Layout
    ------
    Main panel : 28-block × 16-sublayer heatmap.
      - Colormap is RdBu_r centred at 1.0 via TwoSlopeNorm:
          deep red  = cut / heavily reduced
          white     = unchanged (scale ≈ 1.0)
          deep blue = amplified (scale > 1.0)
      - Cut layers (scale == 0.0) are overlaid with a dark hatched cell
        so they are visually distinct from merely-low-scaled layers.
      - Each cell is annotated with its numeric scale value.
    Right strip: per-block bar chart showing the fraction of cut layers.
    Bottom strip: per-sublayer-type bar chart showing mean scale.
    Colorbar on the far right.

    The PNG is saved to <output_path with .safetensors replaced by _scales.png>.
    Requires matplotlib; a warning is printed if it is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        print("  matplotlib not found — skipping plot.  pip install matplotlib")
        return

    import re

    N_COLS = len(_ALL_SUBLAYERS)

    # ------------------------------------------------------------------
    # Build the scale matrix [n_blocks, n_cols]
    # ------------------------------------------------------------------
    matrix = np.full((_N_BLOCKS, N_COLS), np.nan)
    for lora_name, scale in scales.items():
        m = re.match(r"lora_unet_blocks_(\d+)_(.+)$", lora_name)
        if not m:
            continue
        b = int(m.group(1))
        s = m.group(2)
        if b >= _N_BLOCKS or s not in _ALL_SUBLAYERS:
            continue
        matrix[b, _ALL_SUBLAYERS.index(s)] = scale

    cut_mask    = (matrix == 0.0)           # exact zero = ablation cut
    display     = matrix.copy()
    display[cut_mask] = np.nan              # exclude cuts from colormap range

    present = ~np.isnan(matrix)
    n_total = int(present.sum())
    n_cut   = int(cut_mask.sum())
    mean_sc = float(np.nanmean(display)) if n_total > n_cut else 0.0

    # ------------------------------------------------------------------
    # Colormap: RdBu_r centred at 1.0
    # ------------------------------------------------------------------
    valid = display[~np.isnan(display)]
    vmin  = float(np.nanmin(valid)) if valid.size else 0.0
    vmax  = float(np.nanmax(valid)) if valid.size else 2.0
    vmin  = min(vmin, 0.3)   # always show some red range
    vmax  = max(vmax, 1.2)   # always show some blue range
    norm  = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap  = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e0e0e0")   # NaN (missing layer) → light grey

    # ------------------------------------------------------------------
    # Figure layout: heatmap | cut-per-block bars | colorbar
    #                                    mean-per-sublayer bars below
    # ------------------------------------------------------------------
    cell_w = 0.75
    cell_h = 0.40
    fig_w  = N_COLS * cell_w + 4.5
    fig_h  = _N_BLOCKS * cell_h + 3.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[N_COLS, 2.5, 0.5],
        height_ratios=[_N_BLOCKS, 2.5],
        hspace=0.08,
        wspace=0.06,
    )
    ax_heat  = fig.add_subplot(gs[0, 0])   # main heatmap
    ax_rbar  = fig.add_subplot(gs[0, 1])   # right: cut fraction per block
    ax_cbar  = fig.add_subplot(gs[0, 2])   # colorbar
    ax_bbar  = fig.add_subplot(gs[1, 0])   # bottom: mean scale per sublayer

    # ------------------------------------------------------------------
    # Main heatmap
    # ------------------------------------------------------------------
    im = ax_heat.imshow(
        display, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest",
    )

    # Cut cells: dark hatched overlay
    for r in range(_N_BLOCKS):
        for c in range(N_COLS):
            if cut_mask[r, c]:
                ax_heat.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.5, r - 0.5), 1.0, 1.0,
                    boxstyle="square,pad=0",
                    linewidth=0.3, edgecolor="#666",
                    facecolor="#1c1c1c", hatch="////", alpha=0.88,
                    zorder=3,
                ))

    # Cell text annotations
    fontsize_cell = max(3.5, min(5.5, 72 / max(N_COLS, _N_BLOCKS)))
    for r in range(_N_BLOCKS):
        for c in range(N_COLS):
            val = matrix[r, c]
            if np.isnan(val):
                continue
            if val == 0.0:
                ax_heat.text(c, r, "cut", ha="center", va="center",
                             fontsize=fontsize_cell, color="#ffffff",
                             fontweight="bold", zorder=4)
            else:
                # Pick text colour for legibility against the cell background
                normed = norm(val)
                bg_lum = 0.2126 * cmap(normed)[0] + 0.7152 * cmap(normed)[1] + 0.0722 * cmap(normed)[2]
                txt_c  = "black" if bg_lum > 0.45 else "white"
                ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
                             fontsize=fontsize_cell, color=txt_c, zorder=4)

    # Grid lines
    for x in range(N_COLS + 1):
        ax_heat.axvline(x - 0.5, color="white", linewidth=0.4, alpha=0.5)
    for y in range(_N_BLOCKS + 1):
        ax_heat.axhline(y - 0.5, color="white", linewidth=0.4, alpha=0.5)

    # Column group separators (thicker white lines between groups)
    col_cursor = 0
    group_mid_x: list[tuple[float, str]] = []
    for grp_label, subs in _COL_GROUPS:
        group_mid_x.append((col_cursor + len(subs) / 2 - 0.5, grp_label))
        if col_cursor > 0:
            ax_heat.axvline(col_cursor - 0.5, color="white", linewidth=2.0)
        col_cursor += len(subs)

    # Tick labels: abbreviate to the last two underscore-parts of each sublayer name
    # e.g. "self_attn_q_proj" → "q_proj",  "adaln_modulation_mlp_1" → "mlp_1"
    short_col = [
        s.split("_")[-2] + "_" + s.split("_")[-1]
        for s in _ALL_SUBLAYERS
    ]

    ax_heat.set_xticks(range(N_COLS))
    ax_heat.set_xticklabels(short_col, fontsize=6, rotation=45, ha="right")
    ax_heat.set_yticks(range(_N_BLOCKS))
    ax_heat.set_yticklabels([f"B{b:02d}" for b in range(_N_BLOCKS)], fontsize=7)
    ax_heat.set_xlim(-0.5, N_COLS - 0.5)
    ax_heat.set_ylim(_N_BLOCKS - 0.5, -0.5)

    # Group labels on a twin x-axis at the top
    ax_top = ax_heat.twiny()
    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_xticks([mx for mx, _ in group_mid_x])
    ax_top.set_xticklabels([lb for _, lb in group_mid_x],
                            fontsize=8.5, fontweight="bold")
    ax_top.tick_params(length=0)

    ax_heat.set_title(
        f"Per-layer LoRA scales   {n_total} layers  ·  "
        f"{n_cut} cut ({100 * n_cut / max(n_total, 1):.0f}%)  ·  "
        f"mean scale = {mean_sc:.3f}",
        fontsize=10, pad=22,
    )

    # ------------------------------------------------------------------
    # Right strip: fraction of layers cut per block
    # ------------------------------------------------------------------
    block_cut_frac = []
    block_mean_sc  = []
    for b in range(_N_BLOCKS):
        row = matrix[b]
        n_present = int(np.sum(~np.isnan(row)))
        n_cut_row = int(np.sum(row == 0.0))
        block_cut_frac.append(n_cut_row / max(n_present, 1))
        remaining = row[(row != 0.0) & ~np.isnan(row)]
        block_mean_sc.append(float(np.mean(remaining)) if remaining.size else 1.0)

    y_pos = np.arange(_N_BLOCKS)
    bar_colors = [plt.get_cmap("Reds")(0.3 + 0.6 * f) for f in block_cut_frac]
    ax_rbar.barh(y_pos, block_cut_frac, color=bar_colors, height=0.75)
    ax_rbar.set_xlim(0, 1)
    ax_rbar.set_ylim(_N_BLOCKS - 0.5, -0.5)
    ax_rbar.set_yticks([])
    ax_rbar.set_xlabel("Cut fraction", fontsize=7)
    ax_rbar.xaxis.set_tick_params(labelsize=6)
    ax_rbar.axvline(0, color="grey", linewidth=0.5)
    ax_rbar.set_title("Cut\nper block", fontsize=7, pad=4)
    for b, frac in enumerate(block_cut_frac):
        if frac > 0.05:
            ax_rbar.text(frac + 0.01, b, f"{frac:.0%}", va="center",
                         fontsize=5.5, color="#333")

    # ------------------------------------------------------------------
    # Bottom strip: mean scale per sublayer
    # ------------------------------------------------------------------
    col_means = []
    for c in range(N_COLS):
        col_vals = display[:, c]
        col_vals = col_vals[~np.isnan(col_vals)]
        col_means.append(float(np.mean(col_vals)) if col_vals.size else 1.0)

    x_pos    = np.arange(N_COLS)
    bar_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    col_colors = [cmap(bar_norm(v)) for v in col_means]
    bars = ax_bbar.bar(x_pos, col_means, color=col_colors, width=0.75)
    ax_bbar.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bbar.set_xlim(-0.5, N_COLS - 0.5)
    ax_bbar.set_ylim(0, max(max(col_means) * 1.15, 1.2))
    ax_bbar.set_xticks(x_pos)
    ax_bbar.set_xticklabels(short_col, fontsize=6, rotation=45, ha="right")
    ax_bbar.set_ylabel("Mean scale\n(excl. cuts)", fontsize=7)
    ax_bbar.yaxis.set_tick_params(labelsize=6)

    # Group separators on bottom bar chart
    col_cursor = 0
    for _, subs in _COL_GROUPS:
        if col_cursor > 0:
            ax_bbar.axvline(col_cursor - 0.5, color="grey",
                            linewidth=1.2, alpha=0.5)
        col_cursor += len(subs)

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    fig.colorbar(im, cax=ax_cbar, label="Scale factor")
    ax_cbar.yaxis.set_tick_params(labelsize=7)

    # Legend patch for cut cells
    cut_patch = mpatches.Patch(
        facecolor="#1c1c1c", hatch="////", edgecolor="#666",
        label=f"Cut ({n_cut} layers)",
    )
    ax_heat.legend(
        handles=[cut_patch], loc="lower right",
        fontsize=7, framealpha=0.9, borderpad=0.5,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    png_path = output_path.replace(".safetensors", "_scales.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scale plot : {png_path}")


# ---------------------------------------------------------------------------
# Bake scales into LoRA weights and save
# ---------------------------------------------------------------------------

def bake_and_save(
    lora_weights: dict[str, torch.Tensor],
    scales:       dict[str, float],
    output_path:  str,
    dtype:        torch.dtype = torch.bfloat16,
) -> None:
    """Apply final scales to LoRA weight tensors and write a new .safetensors.

    For each layer:
      - scale == 0.0  →  zero both lora_up and lora_down (ablation-determined cut)
      - otherwise     →  multiply lora_up by the learned scale

    All floating-point weight tensors are cast to *dtype* before saving.
    Non-floating-point tensors (alpha scalars stored as integers, etc.) are
    written unchanged.
    Unmapped layers (not in *scales*) are written at scale=1.0 but still
    recast to *dtype*.
    The alpha tensor is always preserved to keep the LoRA spec valid.
    A CSV of all per-layer scales is saved alongside the output file.
    """
    base_keys = {
        k[: -len(".lora_down.weight")]
        for k in lora_weights
        if k.endswith(".lora_down.weight")
    }

    cut_n   = 0
    scale_n = 0
    keep_n  = 0
    rows: list[tuple[str, float]] = []

    # Build output dict: recast every floating-point tensor to the target dtype
    output = {
        k: (v.to(dtype) if v.is_floating_point() else v)
        for k, v in lora_weights.items()
    }

    for base in sorted(base_keys):
        up_k   = f"{base}.lora_up.weight"
        down_k = f"{base}.lora_down.weight"
        scale  = scales.get(base, 1.0)
        rows.append((base, scale))

        if scale == 0.0:
            # Ablation determined this layer hurts or is neutral — zero it out
            output[up_k]   = torch.zeros_like(output[up_k])
            output[down_k] = torch.zeros_like(output[down_k])
            cut_n += 1
        elif abs(scale - 1.0) > 1e-4:
            output[up_k] = (lora_weights[up_k].float() * scale).to(dtype)
            scale_n += 1
        else:
            keep_n += 1

    rows.sort(key=lambda r: r[1])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(output, output_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"Bake summary — {len(rows)} LoRA layers")
    print(f"  Cut  (ablation-determined)    : {cut_n}")
    print(f"  Rescaled (|scale−1| > 0.0001) : {scale_n}")
    print(f"  Unchanged                     : {keep_n}")
    print(f"{'=' * 62}")

    # Print the 15 lowest-scale layers (cuts and heavy reductions first)
    n_show = min(len(rows), 15)
    print(f"\n  {'Layer (last 5 parts)':<46} {'Scale':>7}")
    print(f"  {'-' * 54}")
    for name, sc in rows[:n_show]:
        short = "_".join(name.split("_")[-5:])
        tag   = "  [CUT]" if sc == 0.0 else ""
        print(f"  {short:<46} {sc:7.4f}{tag}")
    if len(rows) > n_show:
        mid_sc = rows[len(rows) // 2][1]
        max_sc = rows[-1][1]
        print(f"  ... ({len(rows) - n_show} more — median {mid_sc:.4f}, max {max_sc:.4f})")

    print(f"\n  Saved: {output_path}  [{dtype}]")

    # ------------------------------------------------------------------
    # CSV (full list, sorted ascending by scale)
    # ------------------------------------------------------------------
    csv_path = output_path.replace(".safetensors", "_scales.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "scale", "cut"])
        for name, sc in rows:
            w.writerow([name, f"{sc:.8f}", "1" if sc == 0.0 else "0"])
    print(f"  Scale table: {csv_path}")

    plot_scales(scales, output_path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    from library import anima_utils, qwen_image_autoencoder_kl

    dtype = _parse_dtype(args.precision)

    print(f"\nLoading Qwen3 from {args.qwen3_path} ...")
    qwen3_enc, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3_path, dtype=dtype, device="cpu"
    )
    qwen3_enc = qwen3_enc.to(CUDA)
    qwen3_enc.requires_grad_(False)
    qwen3_enc.eval()

    text_model = AnimaTextModel(
        qwen3_enc,
        args.qwen3_path,
        args.t5_tokenizer_path or None,
        device=CUDA,
        dtype=dtype,
    )

    print(f"Loading VAE from {args.vae_path} ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae_path, device="cpu", disable_mmap=True
    )
    vae = vae.to(CUDA, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    print(f"Loading Anima DiT from {args.dit_path} ...")
    dit = anima_utils.load_anima_model(
        device=CUDA,
        dit_path=args.dit_path,
        attn_mode="torch",
        split_attn=False,
        loading_device=CUDA,
        dit_weight_dtype=dtype,
    )
    dit.requires_grad_(False)
    dit.eval()

    for attr in ("pos_embedder", "extra_pos_embedder"):
        emb = getattr(dit, attr, None)
        if emb is not None:
            emb.to(CUDA)

    return dit, vae, text_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimise per-layer LoRA scale factors using training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--lora_file",  required=True,
                        help="Input LoRA .safetensors file to tune.")
    parser.add_argument("--data_dir",   required=True,
                        help="Directory with image + .txt caption pairs (the training data).")
    parser.add_argument("--dit_path",   required=True,
                        help="Anima DiT weights (.safetensors).")
    parser.add_argument("--vae_path",   required=True,
                        help="Anima VAE weights (.safetensors).")
    parser.add_argument("--qwen3_path", required=True,
                        help="Qwen3 text encoder directory.")

    # Optional model
    parser.add_argument("--t5_tokenizer_path", default="",
                        help="T5 tokenizer directory (optional).")
    parser.add_argument("--precision", default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model precision (default: bfloat16).")

    # Data
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize images to this square size before VAE encoding "
                             "(default: 512).  Should match your training resolution.")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of data held out for validation (default: 0.15). "
                             "Set to 0 to use all images for both training and validation "
                             "(no split — val loss equals train loss).")

    # Optimisation
    parser.add_argument("--steps",        type=int,   default=50,
                        help="Number of Adam steps (default: 50).")
    parser.add_argument("--lr",           type=float, default=0.05,
                        help="Adam learning rate for the scale parameters (default: 0.05).")
    parser.add_argument("--warmup_steps", type=float, default=0.0,
                        help="LR warmup applied at the start of optimisation (default: 0 = off).  "
                             "LR ramps linearly from 0 → --lr over the warmup period, then "
                             "holds constant.  "
                             "Supply an integer ≥ 1 for an absolute step count (e.g. 10), "
                             "or a float in (0, 1) for a fraction of --steps "
                             "(e.g. 0.1 = 10%% of steps).  "
                             "Recommended: 0.1–0.2 when using perceptual metrics whose "
                             "gradient signal is noisy at the start of training.")
    parser.add_argument("--batch_size",   type=int,   default=4,
                        help="Denoising samples drawn per step (default: 4).")
    parser.add_argument("--reg_strength", type=float, default=0.01,
                        help="L2 regularisation strength toward scale=1.0 (default: 0.01). "
                             "Set to 0 to allow unconstrained cutting/amplification. "
                             "Raise (e.g. 0.1) to preserve the original LoRA character.")
    parser.add_argument("--patience",     type=int,   default=10,
                        help="Early-stop after this many steps without val improvement "
                             "(default: 10, set to 0 to disable).")
    parser.add_argument("--min_timestep", type=int,   default=0,
                        help="Minimum timestep for denoising samples (default: 0).")
    parser.add_argument("--max_timestep", type=int,   default=1000,
                        help="Maximum timestep for denoising samples (default: 1000).")

    # Output
    parser.add_argument("--output",           default="",
                        help="Output .safetensors path.  Defaults to "
                             "<input_stem>_tuned.safetensors in the same directory.")
    parser.add_argument("--ablation_samples", type=int, default=8,
                        help="Denoising samples used per layer during the ablation cut pass "
                             "(default: 8).  Higher = more accurate cut decisions, slower pass. "
                             "Set to 0 to skip the ablation pass entirely.")
    parser.add_argument("--scale_threshold", type=float, default=0.0,
                        help="Cut any layer (or block) whose learned |scale| is below this "
                             "value after optimisation, before the ablation pass.  Free — no "
                             "forward passes required.  0.0 disables (default).  "
                             "Suggested: 0.05 (cut layers at <5%% strength); "
                             "0.1 for more aggressive trimming.")
    parser.add_argument("--block_level", action="store_true",
                        help="Operate at block granularity (blocks 0-27) instead of individual "
                             "layers.  In optimisation mode, one scale parameter is shared across "
                             "all layers within a block.  In ablation mode, entire blocks are "
                             "zeroed and tested together.  Produces coarser but more robust "
                             "decisions with far fewer parameters to optimise (~28 vs ~300+).")
    parser.add_argument("--skip_optimization", action="store_true",
                        help="Skip the Adam scale-optimisation step entirely.  All scales start "
                             "at 1.0 and the ablation cut pass is run from that baseline.  "
                             "This tests each original LoRA layer's utility directly, which "
                             "produces meaningful cuts even when full optimisation never cuts.  "
                             "Recommended for a fast 'cut-only' pruning run.")
    parser.add_argument("--metric", default="mse",
                        choices=["mse", "lpips", "dists", "gram", "fft", "gram_fft"],
                        help="Loss metric for both optimisation and ablation (default: mse).  "
                             "'mse'      — fast single-step denoising MSE; no extra dependencies.  "
                             "'lpips'    — VGG perceptual distance (pip install lpips); broad "
                             "quality but spatially biased and insensitive to fine texture.  "
                             "'dists'    — structure + texture similarity (pip install piq).  "
                             "'gram'     — multi-scale VGG Gram matrix loss; fully positionally "
                             "invariant; best for stylistic / adverse-image LoRAs (blur, grain, "
                             "glitch, DoF, lens artefacts); no extra install beyond torchvision.  "
                             "'fft'      — 2-D FFT log-power-spectrum L2; perfectly shift-invariant; "
                             "directly captures blur (HF attenuation), grain (HF elevation), "
                             "and lens/compression artefacts; no network, near-zero cost.  "
                             "'gram_fft' — Gram + FFT combined; most comprehensive style metric "
                             "for adverse-image LoRAs; recommended when stylistic fidelity is "
                             "the primary goal.")
    parser.add_argument("--save_previews", action="store_true",
                        help="Save a fully denoised preview image at each new best during "
                             "optimisation and at each accepted cut during ablation.  "
                             "Files are written to the output directory as "
                             "<stem>_<unix_time>_<stage_label>.png so they sort "
                             "chronologically.  Requires the VAE to stay in VRAM.")
    parser.add_argument("--preview_steps", type=int, default=20,
                        help="Number of denoising steps for preview images (default: 20). "
                             "Higher = better quality but slower.  Set to 0 to disable previews.")
    parser.add_argument("--preview_sampler", default="euler_ancestral",
                        choices=["euler", "euler_ancestral", "er_sde"],
                        help="Sampler used for preview denoising (default: euler_ancestral).  "
                             "'euler' — deterministic ODE step, fast and stable.  "
                             "'euler_ancestral' / 'er_sde' — stochastic: reconstructs x0 at "
                             "each step then re-noises to the next sigma level; produces "
                             "more natural textures and is the recommended Anima sampler.")
    parser.add_argument("--preview_schedule", default="beta",
                        choices=["uniform", "beta"],
                        help="Timestep schedule for preview denoising (default: beta).  "
                             "'uniform' — linearly-spaced timesteps.  "
                             "'beta' — Beta(0.6,0.6)-distributed timesteps, denser near "
                             "t≈0.5; matches ComfyUI beta schedule for FLUX-family models. "
                             "Requires scipy; falls back to uniform if unavailable.")
    parser.add_argument("--preview_prompt", default="",
                        help="Positive prompt for preview image generation.  "
                             "Leave blank (default) to use the first training caption.")
    parser.add_argument("--preview_negative_prompt", default="",
                        help="Negative prompt for preview CFG guidance.  "
                             "Only used when --preview_cfg_scale > 1.0.  "
                             "Leave blank for an unconditional (empty) negative.")
    parser.add_argument("--preview_cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale for previews (default: 1.0 = off).  "
                             "Set to e.g. 5.0 to enable CFG; runs two DiT forward passes per "
                             "step (cond + uncond).  Recommended range: 3.5–7.0 for Anima.")
    parser.add_argument("--module_filter",    default="",
                        help="Layer name filter applied when building the DiT layer map "
                             "(default: all layers).  E.g. '!adaln_modulation' to skip adaln.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not os.path.isfile(args.lora_file):
        parser.error(f"LoRA file not found: {args.lora_file}")
    if not os.path.isdir(args.data_dir):
        parser.error(f"Data directory not found: {args.data_dir}")
    if args.min_timestep >= args.max_timestep:
        parser.error("--min_timestep must be less than --max_timestep")
    if not (0.0 <= args.val_fraction < 1.0):
        parser.error("--val_fraction must be in [0, 1) — use 0 for no split")

    if not args.output:
        stem       = Path(args.lora_file).stem
        args.output = str(Path(args.lora_file).parent / f"{stem}_tuned.safetensors")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    dit, vae, text_model = load_models(args)

    # ------------------------------------------------------------------
    # Load LoRA
    # ------------------------------------------------------------------
    print(f"\nLoading LoRA from {args.lora_file} ...")
    lora_weights  = load_file(args.lora_file)
    n_lora_layers = sum(1 for k in lora_weights if k.endswith(".lora_down.weight"))
    print(f"  {n_lora_layers} LoRA layers found")

    # ------------------------------------------------------------------
    # Scan and encode dataset
    # ------------------------------------------------------------------
    pairs = scan_data_dir(args.data_dir)
    if not pairs:
        print(f"Error: no image+caption pairs found in {args.data_dir}")
        sys.exit(1)
    print(f"\nFound {len(pairs)} image-caption pairs")

    dtype   = _parse_dtype(args.precision)
    records = encode_dataset(pairs, vae, text_model, args.image_size, dtype)

    # VAE is no longer needed for encoding — free VRAM unless LPIPS needs it
    needs_vae = (args.metric != "mse") or (args.save_previews and args.preview_steps > 0)
    if not needs_vae:
        del vae
        vae = None
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Train / val split (fixed seed for reproducibility)
    # ------------------------------------------------------------------
    if args.val_fraction == 0.0:
        train_data = records
        val_data   = records
        print(f"\nNo split: all {len(records)} images used for both train and val")
    else:
        random.seed(42)
        indices = list(range(len(records)))
        random.shuffle(indices)
        n_val      = max(1, int(len(records) * args.val_fraction))
        val_idx    = set(indices[:n_val])
        val_data   = [records[i] for i in range(len(records)) if     i in val_idx]
        train_data = [records[i] for i in range(len(records)) if i not in val_idx]
        print(f"\nSplit: {len(train_data)} train / {len(val_data)} val")

    # ------------------------------------------------------------------
    # Build layer map and install scale hooks
    # ------------------------------------------------------------------
    layer_map = build_lora_layer_map(dit, module_filter=args.module_filter)
    print(f"DiT exposes {len(layer_map)} LoRA-targetable layers")

    hooks     = LoRAScaleHooks(lora_weights, layer_map, block_level=args.block_level)
    scheduler = AnimaFlowScheduler()

    # ------------------------------------------------------------------
    # Perceptual metric setup (load once if metric != mse, used in both stages)
    # ------------------------------------------------------------------
    perceptual_fn = None
    if args.metric != "mse":
        _labels = {
            "lpips":    "LPIPS VGG network",
            "dists":    "DISTS network",
            "gram":     "VGG Gram matrix loss",
            "fft":      "FFT power-spectrum loss",
            "gram_fft": "VGG Gram + FFT loss",
        }
        print(f"\nLoading {_labels.get(args.metric, args.metric)} ...")
        perceptual_fn = _load_perceptual(args.metric, device=CUDA)

    preview_fn = make_preview_fn(dit, vae, val_data, args, args.output, text_model=text_model)
    if preview_fn:
        print(f"Preview images enabled ({args.preview_steps} denoising steps) → {Path(args.output).parent}")

    # ------------------------------------------------------------------
    # Optimise + ablate (KeyboardInterrupt saves whatever exists so far)
    # ------------------------------------------------------------------
    best_scales  = None
    final_scales = None
    interrupted  = False
    try:
        if args.skip_optimization:
            print("\n[skip_optimization] Skipping Adam step — all scales fixed at 1.0.")
            print(f"Ablation will cut against the original LoRA (scale=1.0) baseline [{args.metric.upper()}].")
            best_scales = {base: 1.0 for base in hooks.scales}
            for base in hooks.scales:
                hooks.scales[base].data.fill_(1.0)
            hooks.install(layer_map)
            fwd_ctx_bl = _FwdCtx(dit)
            n_bl     = max(args.batch_size, args.ablation_samples or args.batch_size)
            dtype_bl = _parse_dtype(args.precision)
            with torch.no_grad():
                if args.metric != "mse":
                    best_val_loss = compute_perceptual_loss(
                        fwd_ctx_bl, scheduler, val_data, n_bl, dtype_bl,
                        args.min_timestep, args.max_timestep, vae, perceptual_fn,
                    ).item()
                else:
                    best_val_loss = compute_loss(
                        fwd_ctx_bl, scheduler, val_data, n_bl, dtype_bl,
                        args.min_timestep, args.max_timestep,
                    ).item()
            hooks.remove()
            print(f"  Baseline ({args.metric.upper()}, scale=1.0): {best_val_loss:.5f}")
        else:
            best_scales, best_val_loss = run_optimization(
                dit, scheduler, hooks, layer_map, train_data, val_data, args,
                vae=vae, perceptual_fn=perceptual_fn, preview_fn=preview_fn,
            )

        # ------------------------------------------------------------------
        # Threshold cut: zero any layer/block whose |scale| < threshold.
        # Free — no forward passes.  Runs before ablation so the sequential
        # loss tests start from an already-trimmed set.
        # ------------------------------------------------------------------
        if args.scale_threshold > 0.0:
            n_thresh = sum(
                1 for v in best_scales.values()
                if abs(v) < args.scale_threshold and v != 0.0
            )
            if n_thresh:
                best_scales = {
                    k: (0.0 if abs(v) < args.scale_threshold else v)
                    for k, v in best_scales.items()
                }
                print(f"\nThreshold cut (|α| < {args.scale_threshold}): "
                      f"{n_thresh} {'blocks' if args.block_level else 'layers'} zeroed")
                if preview_fn:
                    preview_fn("threshold_cut")
            else:
                print(f"\nThreshold cut (|α| < {args.scale_threshold}): nothing to cut")

        if args.ablation_samples > 0:
            final_scales = ablation_cut_pass(
                dit, scheduler, hooks, layer_map,
                val_data, best_scales, best_val_loss, args,
                vae=vae, perceptual_fn=perceptual_fn, preview_fn=preview_fn,
            )
        else:
            print("\nAblation pass skipped (--ablation_samples 0).")
            final_scales = best_scales

    except KeyboardInterrupt:
        interrupted  = True
        final_scales = final_scales or best_scales
        print("\n\n[Interrupted]")

    # ------------------------------------------------------------------
    # Bake and save
    # ------------------------------------------------------------------
    scales_to_save = final_scales or best_scales
    if scales_to_save is None:
        print("No scales computed — nothing to save.")
        sys.exit(1)

    if interrupted:
        stem             = Path(args.output).stem
        interrupted_path = str(Path(args.output).parent / f"{stem}_interrupted.safetensors")
        print(f"Saving interrupted result to: {interrupted_path}")
        bake_and_save(lora_weights, scales_to_save, interrupted_path, dtype=_parse_dtype(args.precision))
    else:
        bake_and_save(lora_weights, scales_to_save, args.output, dtype=_parse_dtype(args.precision))

    print("\nDone." if not interrupted else "\nInterrupted — partial result saved.")


if __name__ == "__main__":
    main()
