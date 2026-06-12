#!/usr/bin/env python
"""
Visualise what each loss component "sees" — how images are transformed
and what features are extracted.

Usage
-----
::

    python diagnostics/visualise_loss_components.py --image path/to/image.jpg

    # Compare two different images
    python diagnostics/visualise_loss_components.py \\
        --image path/to/image_a.jpg \\
        --image-b path/to/image_b.jpg

    # With VAE encoding for realistic latent-space view
    python diagnostics/visualise_loss_components.py \\
        --image path/to/image.jpg \\
        --vae path/to/anima-vae.safetensors

The script produces a side-by-side montage showing each component's
feature extraction and loss maps.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Loss component imports (CPU-compatible, no VAE needed for basic mode)
# ---------------------------------------------------------------------------
from trainer.loss_components.sobel_gradient import SobelGradientLoss, _edge_magnitude
from trainer.loss_components.ms_ssim import MSSSIMLoss, _ssim_map
from trainer.loss_components.census import CensusLoss, _census_hamming
from trainer.loss_components.channel_weight import ChannelWeightLoss
from trainer.loss_components.spatial_highpass import SpatialHighPassLoss


# ===========================================================================
# Image I/O
# ===========================================================================

def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load image → ``[1, 3, H, W]`` float32 in ``[0, 1]``."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert ``[1, C, H, W]`` in ``[0, 1]`` to PIL RGB."""
    t = t.detach().cpu().float().clamp(0, 1)
    if t.shape[1] == 1:
        t = t.expand(-1, 3, -1, -1)
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def latent_to_heatmap(latent: torch.Tensor, channel: Optional[int] = None) -> Image.Image:
    """Visualise latent channel(s) as a heatmap."""
    t = latent.detach().cpu().float()
    if channel is not None:
        t = t[:, channel:channel+1, :, :]
    else:
        # Mean across channels
        t = t.mean(dim=1, keepdim=True)
    # Normalise to [0, 1]
    t = t - t.min()
    t = t / (t.max() + 1e-8)
    # Apply colour map: grayscale → repeated to RGB
    arr = (t.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


# ===========================================================================
# Create test image pairs
# ===========================================================================

def make_test_pairs(img_a: torch.Tensor) -> dict:
    """Generate a family of image variants to test loss component behaviour.

    Returns dict of ``name → [1, 3, H, W]`` tensors.
    """
    pairs = {"original": img_a.clone()}

    # 1. Greyscale (colour removed, structure preserved)
    grey = img_a.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    pairs["greyscale"] = grey

    # 2. Hue shift (colour changed, structure preserved)
    from torchvision.transforms.functional import adjust_hue
    pairs["hue_shift"] = adjust_hue(img_a, 0.3)

    # 3. Saturation reduced
    from torchvision.transforms.functional import adjust_saturation
    pairs["low_saturation"] = adjust_saturation(img_a, 0.2)

    # 4. Spatially shifted (translate 10px)
    shift = 10
    shifted = torch.roll(img_a, shifts=(shift, shift), dims=(2, 3))
    pairs["shifted"] = shifted

    # 5. Blurred (structure removed, colour preserved)
    kernel = torch.ones(1, 1, 15, 15) / 225.0
    blurred = F.conv2d(
        F.pad(img_a, (7, 7, 7, 7), mode="reflect"),
        kernel.expand(3, 1, 15, 15), groups=3
    )
    pairs["blurred"] = blurred

    # 6. Edge-enhanced
    from torchvision.transforms.functional import adjust_sharpness
    pairs["sharpened"] = adjust_sharpness(img_a, 2.0)

    # 7. Brightness shifted
    from torchvision.transforms.functional import adjust_brightness
    pairs["brightness_shift"] = adjust_brightness(img_a, 1.5)

    return pairs


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def make_label(text: str, size: Tuple[int, int] = (256, 30)) -> Image.Image:
    """Create a label image with text."""
    img = Image.new("RGB", size, (30, 30, 30))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    # Centre text
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) // 2
    y = (size[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=(220, 220, 220), font=font)
    return img


def make_montage(images: dict, cols: int = 4) -> Image.Image:
    """Arrange labelled images in a grid."""
    items = list(images.items())
    rows = (len(items) + cols - 1) // cols
    # Get sample size
    sample = list(images.values())[0]
    iw, ih = sample.size
    label_h = 30

    montage = Image.new(
        "RGB", (cols * iw, rows * (ih + label_h)), (20, 20, 20)
    )

    for idx, (name, img) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x = c * iw
        y = r * (ih + label_h)

        label = make_label(name, (iw, label_h))
        montage.paste(label, (x, y))
        montage.paste(img, (x, y + label_h))

    return montage


# ===========================================================================
# Main visualisation
# ===========================================================================

def visualise_loss_components(
    image_path: str,
    image_path_b: Optional[str] = None,
    vae_path: Optional[str] = None,
    output_path: str = "loss_components_visualisation.png",
    image_size: int = 512,
    tile_size: int = 256,
):
    """Run the full visualisation pipeline and save a montage."""
    print(f"Loading image: {image_path}")
    img_a = load_image(image_path, image_size)

    # Use VAE if provided to show latent-space view
    if vae_path:
        print(f"Loading VAE: {vae_path}")
        from trainer.qwen_image_autoencoder_kl import load_vae
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = load_vae(vae_path, device="cpu", disable_mmap=True)
        vae = vae.to(device, dtype=torch.bfloat16)
        vae.eval()

        with torch.no_grad():
            latent_a = vae.encode_pixels_to_latents(
                (img_a * 2.0 - 1.0).to(device)
            )
        pixel_domain = False
        print("  Using VAE latent space (16 channels)")
    else:
        # Use pixel space directly (simulates single-channel latent behaviour)
        latent_a = img_a.mean(dim=1, keepdim=True)  # [1, 1, H, W]
        pixel_domain = True
        device = torch.device("cpu")
        print("  Using pixel luminance (no VAE — showing pixel-space view)")

    latent_a = latent_a.float()

    # Build test pairs in latent space
    pairs = {}
    pairs["original"] = latent_a.clone()

    # Greyscale in latent space
    if pixel_domain:
        pairs["greyscale"] = latent_a.clone()  # already grey
    elif vae_path:
        img_grey = img_a.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        with torch.no_grad():
            latent_grey = vae.encode_pixels_to_latents(
                (img_grey * 2.0 - 1.0).to(device)
            ).float()
        pairs["greyscale"] = latent_grey

    # Shifted
    shift = 4  # latent pixels
    pairs["shifted_4px"] = torch.roll(latent_a, shifts=(shift, shift), dims=(2, 3))

    # Noise
    pairs["noisy"] = latent_a + torch.randn_like(latent_a) * 0.1

    # Colour-channel ablation if VAE: zero out different channels
    if not pixel_domain:
        for ch in range(min(4, latent_a.shape[1])):
            modified = latent_a.clone()
            modified[:, ch, :, :] = 0
            pairs[f"zero_ch_{ch}"] = modified

    # ---- Compute and visualise each loss component ----

    results = {}

    # Helper: compute loss map for a variant vs original
    def compute_loss_map(component, variant_name, **kwargs) -> Image.Image:
        variant = pairs.get(variant_name, latent_a)
        with torch.no_grad():
            if hasattr(component, 'forward'):
                if isinstance(component, CensusLoss):
                    # Census uses custom compute function
                    loss_map = _census_hamming(
                        variant, latent_a,
                        component._radius
                    )
                elif isinstance(component, MSSSIMLoss):
                    loss_map = 1.0 - _ssim_map(
                        variant, latent_a,
                        component._window_size,
                        component._sigma,
                        component._structure_only
                    )
                elif isinstance(component, SobelGradientLoss):
                    edges_v = _edge_magnitude(variant)
                    edges_o = _edge_magnitude(latent_a)
                    loss_map = (edges_v - edges_o).abs()
                else:
                    # Generic forward
                    loss_map = component(variant, latent_a)
            else:
                loss_map = component(variant, latent_a)

        loss_map = loss_map.float()
        # Aggregate channels if multi-channel
        if loss_map.shape[1] > 1:
            loss_map = loss_map.mean(dim=1, keepdim=True)

        # Normalise to 0-1 for display
        loss_map = loss_map - loss_map.min()
        loss_map = loss_map / (loss_map.max() + 1e-8)
        return latent_to_heatmap(loss_map)

    # ---- 1. Sobel edge visualisation ----
    print("  Sobel edges...")
    sobel = SobelGradientLoss(normalise_edges=True)
    edges = _edge_magnitude(latent_a)
    if not pixel_domain:
        edges = edges.mean(dim=1, keepdim=True)
    edges_img = latent_to_heatmap(edges)
    results["sobel_edges"] = edges_img

    # Sobel loss for each variant
    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs:
            results[f"sobel_loss_{vname}"] = compute_loss_map(sobel, vname)

    # ---- 2. MS-SSIM visualisation ----
    print("  MS-SSIM structure maps...")
    ssim = MSSSIMLoss(window_size=11, sigma=1.5, scales=1, use_structure_only=True)

    results["ssim_self"] = compute_loss_map(ssim, "original")
    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs:
            results[f"ssim_loss_{vname}"] = compute_loss_map(ssim, vname)

    # Multi-scale: show comparison
    ssim_ms = MSSSIMLoss(window_size=7, sigma=1.5, scales=3, use_structure_only=True)
    results["ssim_ms_shifted"] = compute_loss_map(ssim_ms, "shifted_4px")

    # ---- 3. Census visualisation ----
    print("  Census Hamming distance...")
    census = CensusLoss(window_size=3)

    results["census_self"] = compute_loss_map(census, "original")
    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs:
            results[f"census_loss_{vname}"] = compute_loss_map(census, vname)

    census5 = CensusLoss(window_size=5)
    results["census_5x5_shifted"] = compute_loss_map(census5, "shifted_4px")

    # ---- 4. Channel weight visualisation ----
    print("  Channel weighting...")
    # Generate diagnostic-style channel weights
    C = latent_a.shape[1]
    fake_weights = torch.ones(C)
    if not pixel_domain:
        # Simulate: some channels get low weight
        fake_weights[1] = 0.2
        fake_weights[3] = 0.3
        fake_weights[5] = 0.1
    cw_vis = ChannelWeightLoss(weights=fake_weights)

    # Show per-channel weight bar
    weight_bar = torch.zeros(1, C, 1, 1)
    for ch in range(C):
        weight_bar[0, ch, 0, 0] = fake_weights[ch].item() * 255
    # Resize to visible
    weight_bar = weight_bar.expand(-1, -1, C * 8, 64)
    results["channel_weights"] = latent_to_heatmap(weight_bar)

    # Weighted loss maps for variants
    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs:
            results[f"weighted_loss_{vname}"] = compute_loss_map(cw_vis, vname)

    # PCA projection visualisation
    if not pixel_domain:
        # Fake PCA: first N channels are "colour subspace"
        fake_eigvecs = torch.eye(C)
        fake_n_colour = 3
        fake_analysis = {
            "colour_subspace_eigvecs": fake_eigvecs,
            "n_colour_dimensions": fake_n_colour,
            "channel_weights": torch.ones(C),
        }
        cw_pca = ChannelWeightLoss(analysis_dict=fake_analysis, mode="pca_projection")
        results["pca_shifted"] = compute_loss_map(cw_pca, "shifted_4px")
        results["pca_greyscale"] = compute_loss_map(cw_pca, "greyscale")

    # ---- 5. Spatial high-pass visualisation ----
    print("  Spatial high-pass...")
    hpf = SpatialHighPassLoss(strength=2.0, kernel_size=5, sigma=2.0)

    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs:
            results[f"hpf_loss_{vname}"] = compute_loss_map(hpf, vname)

    # ---- 6. Raw image comparison ----
    print("  Reference images...")
    if pixel_domain:
        results["input_greyscale"] = tensor_to_pil(latent_a)
    else:
        results["input_rgb"] = tensor_to_pil(img_a)

    for vname in ["greyscale", "shifted_4px", "noisy"]:
        if vname in pairs and pixel_domain:
            results[f"variant_{vname}"] = tensor_to_pil(pairs[vname])
        elif vname in pairs and not pixel_domain and vname == "greyscale":
            with torch.no_grad():
                dec = vae.decode_to_pixels(pairs[vname].to(device))
            results[f"variant_{vname}"] = tensor_to_pil((dec * 0.5 + 0.5).clamp(0, 1))

    # ---- Assemble and save ----
    print(f"\nAssembling montage ({len(results)} panels)...")
    montage = make_montage(results, cols=4)
    montage.save(output_path)
    print(f"Saved to {output_path}")

    # Print legend
    print("\n" + "=" * 60)
    print("LEGEND — What each panel shows")
    print("=" * 60)
    print("  sobel_edges         — Edge magnitudes of the input image")
    print("  sobel_loss_X        — Where Sobel detects differences between input and X")
    print("  ssim_self           — SSIM self-comparison (should be black = perfect)")
    print("  ssim_loss_X         — Where SSIM detects structural differences from X")
    print("  ssim_ms_shifted     — Multi-scale SSIM (3 scales) response to translation")
    print("  census_self         — Census self-comparison (should be black)")
    print("  census_loss_X       — Where neighbour-relations differ from X")
    print("  census_5x5_shifted  — Larger 5×5 census window response to translation")
    print("  channel_weights     — Per-channel weight colour bar")
    print("  weighted_loss_X     — Channel-weighted loss for variant X")
    print("  pca_*               — PCA projection loss (structure subspace only)")
    print("  hpf_loss_X          — Spatial high-pass emphasised loss for X")
    print("\nInterpretation:")
    print("  Bright regions = large loss = the component 'cares about' this difference")
    print("  Dark regions = small loss = the component considers this 'close enough'")
    print("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise what each loss component extracts from images."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--image-b", default=None,
        help="Optional second image for cross-comparison"
    )
    parser.add_argument(
        "--vae", default=None,
        help="Path to Anima VAE .safetensors (latent-space view)"
    )
    parser.add_argument(
        "--output", default="loss_components_visualisation.png",
        help="Output image path"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Resize images to this size (default: 512)"
    )
    args = parser.parse_args()

    visualise_loss_components(
        image_path=args.image,
        image_path_b=args.image_b,
        vae_path=args.vae,
        output_path=args.output,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
