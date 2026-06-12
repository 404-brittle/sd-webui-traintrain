#!/usr/bin/env python
"""
VAE latent-channel sensitivity analysis — **richer diagnostic**.

Measures how colour information is distributed across the 16 Anima VAE latent
channels, producing a comprehensive analysis file consumable by the
:class:`~trainer.loss_components.channel_weight.ChannelWeightLoss`
component.

Output ``.pt`` file structure
-----------------------------
A dictionary with the following keys:

``channel_weights`` ``[16]``
    Simple per-channel multipliers in ``(0, 1]``.  Channels with *large*
    mean-abs-diff between colour and greyscale encodings get *low* weight.
    This is the backward-compatible weight vector for simple use.

``channel_diff_mean`` ``[16]``
    Mean absolute latent difference ``|rgb - grey|`` per channel (averaged
    over all spatial positions and images).

``channel_diff_std`` ``[16]``
    Standard deviation of the absolute difference per channel — captures
    how *consistent* the colour sensitivity is across images/positions.

``channel_covariance`` ``[16, 16]``
    Covariance matrix of the 16 channel responses to colour difference.
    The eigenvectors of this matrix define the **colour subspace** of the
    latent space — directions most affected by hue/saturation changes.

``colour_subspace_eigvals`` ``[16]``
    Eigenvalues of the channel covariance (sorted descending).  Large
    eigenvalues = directions strongly modulated by colour.

``colour_subspace_eigvecs`` ``[16, 16]``
    Corresponding eigenvectors (columns).  Project a latent vector onto
    the trailing eigenvectors to get a colour-invariant representation.

``n_colour_dimensions`` ``int``
    Number of eigenvectors needed to explain 95 % of colour variance.
    The complementary subspace (dimensions ``n_colour_dimensions .. 15``)
    is the **structure subspace** — nearly colour-invariant.

``spatial_freq_profile`` ``[16, num_scales]``
    Per-channel colour sensitivity at each spatial scale (coarse → fine).
    Channels where colour lives at coarse scales (uniform colour fields)
    can be safely downweighted; channels where colour is at fine scales
    carry colour-edges and are riskier to suppress.

``spatial_uniformity`` ``[16]``
    Ratio ``mean(|diff|) / std(|diff|)`` over spatial positions per channel.
    High → colour difference is a global shift (safe to downweight).
    Low → colour difference is spatially structured (colour edges).

Usage
-----
::

    python diagnostics/channel_sensitivity.py \\
        --vae /path/to/anima-vae.safetensors \\
        --data /path/to/training/images

    # Limit to 50 images for a quick check
    python diagnostics/channel_sensitivity.py \\
        --vae /path/to/anima-vae.safetensors \\
        --data /path/to/training/images \\
        --max-samples 50
"""

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Path setup  —  allows running as script from project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------


def _load_vae(vae_path: str, device: torch.device):
    """Load the Anima VAE in eval mode."""
    from trainer.qwen_image_autoencoder_kl import load_vae

    print(f"Loading VAE from {vae_path}")
    vae = load_vae(vae_path, device="cpu", disable_mmap=True)
    vae = vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _find_images(data_path: str) -> List[str]:
    """Recursively collect image paths."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(data_path, "**", ext), recursive=True))
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _rgb_to_greyscale_tensor(img: Image.Image) -> Image.Image:
    """Convert PIL Image to greyscale, then back to 3-channel RGB."""
    grey = img.convert("L")
    return grey.convert("RGB")


def _image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image → ``[1, 3, H, W]`` float32 tensor in ``[-1, 1]``."""
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Spatial frequency decomposition
# ---------------------------------------------------------------------------


def _multiscale_diff(diff: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
    """Decompose a ``[C, H, W]`` absolute-diff tensor into spatial scales.

    Returns ``[C, num_scales]`` where each entry is the mean absolute
    difference at that scale (scale 0 = full resolution, scale N = heavily
    downsampled / coarse).
    """
    C, H, W = diff.shape
    profiles = torch.zeros(C, num_scales, device=diff.device, dtype=diff.dtype)

    x = diff.clone()
    for s in range(num_scales):
        # Mean over spatial dims at this scale
        profiles[:, s] = x.mean(dim=[1, 2])
        # Downsample by 2× for next coarser scale
        if x.shape[1] > 1 and x.shape[2] > 1:
            # Use adaptive avg pool to halve
            x = torch.nn.functional.adaptive_avg_pool2d(x, (max(1, x.shape[1] // 2), max(1, x.shape[2] // 2)))
        else:
            break
    return profiles  # [C, num_scales]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse_channel_sensitivity(
    vae_path: str,
    data_path: str,
    output_path: str = "./weights/ch16.pt",
    max_samples: Optional[int] = None,
    device: str = "cuda:0",
):
    """Run the full multi-faceted analysis pipeline.

    Args:
        vae_path: Path to the Anima VAE ``.safetensors`` file.
        data_path: Path to a directory of training images.
        output_path: Where to save the analysis dictionary (``.pt`` file).
        max_samples: Limit to this many images (``None`` = all).
        device: Torch device string.
    """
    device_t = torch.device(device)
    vae = _load_vae(vae_path, device_t)
    C = vae.z_dim  # number of latent channels (16)
    image_paths = _find_images(data_path)

    if not image_paths:
        print(f"ERROR: no images found under {data_path}")
        sys.exit(1)

    if max_samples is not None and max_samples < len(image_paths):
        print(f"Limiting to {max_samples} images (of {len(image_paths)} found)")
        rng = np.random.RandomState(42)
        image_paths = list(rng.choice(image_paths, max_samples, replace=False))

    print(f"Analysing {len(image_paths)} images across {C} latent channels...\n")

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    # 1. Per-channel mean abs diff
    sum_abs_diff = torch.zeros(C, dtype=torch.float64, device="cpu")
    sum_abs_diff_sq = torch.zeros(C, dtype=torch.float64, device="cpu")

    # 2. Channel covariance (over spatial positions and images)
    # We accumulate (C,) outer products and count spatial positions
    cov_sum = torch.zeros(C, C, dtype=torch.float64, device="cpu")
    cov_count = 0

    # 3. Spatial frequency profiles per channel
    num_scales = 5
    freq_profile_sum = torch.zeros(C, num_scales, dtype=torch.float64, device="cpu")

    # 4. Spatial uniformity: for each channel, sum of mean(|diff|) and sum of std(|diff|)
    uniform_sum_mean = torch.zeros(C, dtype=torch.float64, device="cpu")
    uniform_sum_std = torch.zeros(C, dtype=torch.float64, device="cpu")
    uniform_count = 0

    n_processed = 0

    for path in tqdm(image_paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  Skipping {path}: {exc}")
            continue

        tensor_rgb = _image_to_tensor(img, device_t)          # [1, 3, H, W]
        tensor_grey = _image_to_tensor(
            _rgb_to_greyscale_tensor(img), device_t
        )

        with torch.no_grad():
            latent_rgb = vae.encode_pixels_to_latents(tensor_rgb)    # [1, C, h, w]
            latent_grey = vae.encode_pixels_to_latents(tensor_grey)

        # Signed difference (not absolute!) for covariance analysis
        diff_signed = (latent_rgb.float() - latent_grey.float())     # [1, C, h, w]
        diff_abs = diff_signed.abs()                                 # [1, C, h, w]

        # Squeeze batch dim
        diff_s = diff_signed.squeeze(0)  # [C, h, w]
        diff_a = diff_abs.squeeze(0)     # [C, h, w]

        # ---- 1. Per-channel mean + std of |diff| ----
        ch_mean = diff_a.mean(dim=[1, 2])   # [C]
        ch_var = diff_a.var(dim=[1, 2])     # [C]

        sum_abs_diff += ch_mean.cpu().double()
        sum_abs_diff_sq += (ch_var + ch_mean.pow(2)).cpu().double()  # E[x^2] = Var + (E[x])^2

        # ---- 2. Channel covariance of signed diff ----
        # Flatten spatial dims: [C, h, w] → [C, h*w]
        C_chan, h, w = diff_s.shape
        diff_flat = diff_s.view(C_chan, -1)            # [C, N]  where N = h*w
        diff_centered = diff_flat - diff_flat.mean(dim=1, keepdim=True)  # [C, N]
        cov_sum += (diff_centered @ diff_centered.T).cpu().double()      # [C, C]
        cov_count += diff_flat.shape[1]

        # ---- 3. Spatial frequency profile ----
        freq_profile = _multiscale_diff(diff_a.cpu(), num_scales)  # [C, num_scales]
        freq_profile_sum += freq_profile.double()

        # ---- 4. Spatial uniformity ----
        # mean(|diff|) / std(|diff|) per channel over spatial positions
        spatial_mean = diff_a.mean(dim=[1, 2])           # [C]
        spatial_var = diff_a.var(dim=[1, 2])             # [C]
        uniform_sum_mean += spatial_mean.cpu().double()
        uniform_sum_std += spatial_var.sqrt().cpu().double()
        uniform_count += 1

        n_processed += 1

        del tensor_rgb, tensor_grey, latent_rgb, latent_grey
        del diff_signed, diff_abs, diff_s, diff_a
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

        if max_samples is not None and n_processed >= max_samples:
            break

    # ------------------------------------------------------------------
    # Compute final statistics
    # ------------------------------------------------------------------
    if n_processed == 0:
        print("ERROR: no images could be processed.")
        sys.exit(1)

    # Per-channel statistics
    channel_diff_mean = (sum_abs_diff / n_processed).float()                          # [C]
    # E[|diff|^2] - (E[|diff|])^2, then sqrt for std
    channel_diff_var = (sum_abs_diff_sq / n_processed) - channel_diff_mean.float().pow(2)
    channel_diff_std = channel_diff_var.clamp(min=0).sqrt().float()                    # [C]

    # Channel covariance
    channel_covariance = (cov_sum / max(1, cov_count)).float()                         # [C, C]

    # Eigen decomposition of covariance
    eigvals, eigvecs = torch.linalg.eigh(channel_covariance)
    # torch.linalg.eigh returns ascending; flip to descending
    eigvals = eigvals.flip(dims=[0])
    eigvecs = eigvecs.flip(dims=[1])

    # Number of dims explaining 95 % colour variance
    cumsum = eigvals.cumsum(dim=0)
    total_var = cumsum[-1].clamp(min=1e-12)
    n_colour = int((cumsum / total_var < 0.95).sum().item()) + 1
    n_colour = min(n_colour, C)

    # Simple per-channel weights (backward compat)
    max_diff = channel_diff_mean.max().item()
    if max_diff < 1e-8:
        channel_weights = torch.ones(C, dtype=torch.float32)
    else:
        channel_weights = 1.0 - (channel_diff_mean / max_diff)
        channel_weights = channel_weights.clamp(0.05, 1.0)

    # Spatial frequency profile
    spatial_freq_profile = (freq_profile_sum / n_processed).float()  # [C, num_scales]

    # Spatial uniformity
    # For each channel: mean(mean(|diff|) / std(|diff|)) across images
    # We compute per-image ratio then average (more robust than ratio of averages)
    # Actually compute from accumulated means/stds
    uniform_mean_avg = (uniform_sum_mean / uniform_count).float()    # [C]
    uniform_std_avg = (uniform_sum_std / uniform_count).float()      # [C]
    # Ratio: large = global shift, small = structured
    spatial_uniformity = (uniform_mean_avg / (uniform_std_avg + 1e-8)).float()  # [C]

    # ------------------------------------------------------------------
    # Assemble output dictionary
    # ------------------------------------------------------------------
    result = {
        # Metadata
        "meta": {
            "vae_path": vae_path,
            "n_images": n_processed,
            "latent_channels": C,
            "num_scales": num_scales,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },

        # Simple per-channel weights (backward compat for ChannelWeightLoss)
        "channel_weights": channel_weights,

        # Per-channel difference statistics
        "channel_diff_mean": channel_diff_mean,
        "channel_diff_std": channel_diff_std,

        # Channel covariance and colour subspace
        "channel_covariance": channel_covariance,
        "colour_subspace_eigvals": eigvals,
        "colour_subspace_eigvecs": eigvecs,
        "n_colour_dimensions": n_colour,

        # Spatial frequency decomposition
        "spatial_freq_profile": spatial_freq_profile,

        # Spatial uniformity
        "spatial_uniformity": spatial_uniformity,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(result, output_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHANNEL SENSITIVITY ANALYSIS — SUMMARY")
    print("=" * 70)
    print(f"  Images processed:    {n_processed}")
    print(f"  Latent channels:     {C}")
    print(f"  Output:              {output_path}")
    print(f"  File size:           {os.path.getsize(output_path) / 1024:.1f} kB")
    print()

    # Per-channel table
    print(f"  {'ch':>3s}  {'mean|diff|':>10s}  {'std|diff|':>10s}  {'weight':>7s}  "
          f"{'uniformity':>10s}  {'freq_profile':>20s}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*7}  "
          f"{'-'*10}  {'-'*20}")
    for ch in range(C):
        prof_str = " ".join(f"{spatial_freq_profile[ch, s].item():.3f}" for s in range(num_scales))
        print(f"  {ch:3d}  {channel_diff_mean[ch].item():10.4f}  "
              f"{channel_diff_std[ch].item():10.4f}  "
              f"{channel_weights[ch].item():7.4f}  "
              f"{spatial_uniformity[ch].item():10.4f}  "
              f"{prof_str}")

    print()
    print(f"  Colour-sensitive channels (weight < 0.5): "
          f"{[ch for ch in range(C) if channel_weights[ch].item() < 0.5]}")
    print(f"  Luminance-dominated channels (weight >= 0.5): "
          f"{[ch for ch in range(C) if channel_weights[ch].item() >= 0.5]}")
    print()
    print(f"  Colour subspace PCA:")
    print(f"    Dimensions explaining 95 % variance: {n_colour} / {C}")
    print(f"    Top-5 eigenvalue ratio (λ_i / Σλ):")
    for i in range(min(5, C)):
        print(f"      dim {i}:  {eigvals[i].item():.4f}  "
              f"({(eigvals[i] / total_var * 100).item():.1f} %)")
    print()
    print(f"  Structure subspace dimensions "
          f"(project onto eigvecs[:, {n_colour}:] for colour invariance):")
    print(f"    {list(range(n_colour, C))}")
    print()
    print(f"  Spatial uniformity interpretation:")
    print(f"    High ratio ({'>':s}2.0)  = global colour shift — safe to downweight")
    print(f"    Low ratio  ({'<':s}0.5)  = localized colour edges — risky to downweight")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyse VAE latent-channel sensitivity to colour vs luminance."
    )
    parser.add_argument(
        "--vae",
        required=True,
        help="Path to Anima VAE .safetensors file",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to directory of training images",
    )
    parser.add_argument(
        "--output",
        default="./weights/ch16.pt",
        help="Output path for the analysis .pt file (default: ./weights/ch16.pt)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit analysis to N images (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    args = parser.parse_args()

    analyse_channel_sensitivity(
        vae_path=args.vae,
        data_path=args.data,
        output_path=args.output,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
