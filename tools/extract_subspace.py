#!/usr/bin/env python3
"""Extract a principal subspace from a collection of reference LoRA .safetensors files.

For each LoRA layer the script stacks the lora_down.weight matrices from all
reference LoRAs and computes the top-K right singular vectors via truncated SVD.
These K vectors define the principal input-space directions occupied by that
feature type and are saved to a .safetensors "subspace file" consumed by
trainer/subspace_guard.py during training.

Usage
-----
    # From a directory of reference LoRAs (e.g. style LoRAs):
    python tools/extract_subspace.py \\
        --lora_dir  /path/to/style_loras \\
        --output    subspaces/style.safetensors \\
        --n_components 16

    # From an explicit file list:
    python tools/extract_subspace.py \\
        --lora_files a.safetensors b.safetensors c.safetensors \\
        --output    subspaces/subject.safetensors

    # With singular value decay plot (requires matplotlib):
    python tools/extract_subspace.py --lora_dir ./loras --output sub.safetensors --plot_decay

Output format
-------------
The output .safetensors contains two tensors per layer:
    "<lora_name>.V_K"             — (n, K) fp16  principal input-space directions
    "<lora_name>.singular_values" — (K,)   fp16  corresponding singular values

Only layers present in at least 2 reference LoRAs are included (fewer references
means the subspace estimate is unreliable).
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_subspace(
    lora_paths: list[str],
    n_components: int = 16,
    device: str = "cpu",
    min_references: int = 2,
) -> dict:
    """Compute per-layer principal subspaces from a list of LoRA files.

    Args:
        lora_paths:      List of .safetensors paths.
        n_components:    Number of principal components K to retain per layer.
        device:          Torch device for SVD (use "cuda" for large models).
        min_references:  Layers present in fewer LoRAs than this are skipped.

    Returns:
        dict mapping lora_name (base key, no suffix) to
            {"V_K": Tensor(n, K), "singular_values": Tensor(K)}
    """
    down_weights: dict[str, list[torch.Tensor]] = defaultdict(list)

    print(f"Loading {len(lora_paths)} reference LoRAs...")
    for path in lora_paths:
        try:
            weights = load_file(path, device=device)
        except Exception as exc:
            print(f"  Warning: could not load {path}: {exc}")
            continue

        for key, tensor in weights.items():
            if key.endswith(".lora_down.weight"):
                base = key[: -len(".lora_down.weight")]
                down_weights[base].append(tensor.float())

    n_layers = len(down_weights)
    print(f"Found {n_layers} unique layer keys across reference LoRAs.")

    subspaces: dict = {}
    skipped = 0

    for base_key in sorted(down_weights.keys()):
        matrices = down_weights[base_key]

        if len(matrices) < min_references:
            skipped += 1
            continue

        # Stack all lora_down rows: each (r, n) → (total_r, n)
        stacked = torch.cat(matrices, dim=0).to(device)
        n = stacked.shape[1]
        K = min(n_components, stacked.shape[0] - 1, n - 1)

        if K < 1:
            skipped += 1
            continue

        try:
            # Truncated SVD — only the top-K right singular vectors are needed.
            # stacked = U Σ V^T  →  V^T shape: (min(d*r, n), n)
            _, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            V_K = Vh[:K].T.contiguous()  # (n, K)

            subspaces[base_key] = {
                "V_K": V_K.half().cpu(),
                "singular_values": S[:K].half().cpu(),
            }
        except Exception as exc:
            print(f"  Warning: SVD failed for {base_key}: {exc}")
            skipped += 1
            continue

    print(
        f"Extracted subspaces for {len(subspaces)}/{n_layers} layers "
        f"({skipped} skipped — too few references or SVD failure)."
    )
    return subspaces


# ---------------------------------------------------------------------------
# Save / load helpers (also imported by subspace_guard.py)
# ---------------------------------------------------------------------------

def save_subspace(subspaces: dict, output_path: str) -> None:
    """Flatten and save subspace dict to a .safetensors file."""
    flat: dict[str, torch.Tensor] = {}
    for base_key, data in subspaces.items():
        flat[f"{base_key}.V_K"] = data["V_K"]
        flat[f"{base_key}.singular_values"] = data["singular_values"]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(flat, output_path)
    print(f"Saved subspace ({len(subspaces)} layers) to {output_path}")


def load_subspace(path: str, device: str = "cpu") -> dict:
    """Load a subspace file back into {lora_name: {"V_K": ..., "singular_values": ...}}."""
    flat = load_file(path, device=device)
    subspaces: dict = {}
    for key, tensor in flat.items():
        if key.endswith(".V_K"):
            base = key[: -len(".V_K")]
            subspaces.setdefault(base, {})["V_K"] = tensor
        elif key.endswith(".singular_values"):
            base = key[: -len(".singular_values")]
            subspaces.setdefault(base, {})["singular_values"] = tensor
    return subspaces


# ---------------------------------------------------------------------------
# Singular-value decay plot
# ---------------------------------------------------------------------------

def plot_singular_value_decay(subspaces: dict, output_path: str) -> None:
    """Save a 2×2 panel of singular-value decay curves for a sample of layers."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping decay plot.")
        return

    keys = list(subspaces.keys())
    # Pick 4 representative layers spread across blocks
    if len(keys) >= 4:
        step = len(keys) // 4
        sample_keys = [keys[i * step] for i in range(4)]
    else:
        sample_keys = keys

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Singular value decay (normalised to first component)")

    for ax, key in zip(axes.flat, sample_keys):
        sv = subspaces[key]["singular_values"].float().numpy()
        relative = sv / sv[0]
        ax.semilogy(np.arange(1, len(relative) + 1), relative, "b-o", markersize=3)
        # Short label: last 3 underscore-separated tokens
        label = "_".join(key.split("_")[-4:])
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Relative singular value")
        ax.axhline(0.1, color="r", linestyle="--", alpha=0.5, label="10% threshold")
        ax.legend(fontsize=7)

    # Hide unused panels
    for ax in axes.flat[len(sample_keys):]:
        ax.set_visible(False)

    plt.tight_layout()
    plot_path = os.path.splitext(output_path)[0] + "_decay.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved decay plot to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract principal subspace from a set of reference LoRA .safetensors files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--lora_dir",
        type=str,
        help="Directory containing reference .safetensors LoRA files.",
    )
    src.add_argument(
        "--lora_files",
        nargs="+",
        help="Explicit list of .safetensors LoRA file paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output subspace .safetensors file path.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=16,
        help="Number of principal components K to retain per layer (default: 16).",
    )
    parser.add_argument(
        "--min_references",
        type=int,
        default=2,
        help="Skip layers present in fewer than this many reference LoRAs (default: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for SVD computation. Use 'cuda' for speed (default: cpu).",
    )
    parser.add_argument(
        "--plot_decay",
        action="store_true",
        help="Save a singular-value decay plot alongside the output file (requires matplotlib).",
    )
    args = parser.parse_args()

    # Resolve input paths
    if args.lora_dir:
        if not os.path.isdir(args.lora_dir):
            print(f"Error: --lora_dir '{args.lora_dir}' is not a directory.")
            sys.exit(1)
        lora_paths = [
            os.path.join(args.lora_dir, f)
            for f in sorted(os.listdir(args.lora_dir))
            if f.endswith(".safetensors")
        ]
    else:
        lora_paths = args.lora_files

    if not lora_paths:
        print("Error: no .safetensors files found.")
        sys.exit(1)

    print(f"Reference LoRAs ({len(lora_paths)}):")
    for p in lora_paths:
        print(f"  {p}")

    subspaces = extract_subspace(
        lora_paths,
        n_components=args.n_components,
        device=args.device,
        min_references=args.min_references,
    )

    if not subspaces:
        print("Error: no subspaces could be extracted (check --min_references and input files).")
        sys.exit(1)

    save_subspace(subspaces, args.output)

    if args.plot_decay:
        plot_singular_value_decay(subspaces, args.output)


if __name__ == "__main__":
    main()
