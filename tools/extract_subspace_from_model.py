#!/usr/bin/env python3
"""Extract a principal subspace directly from the Anima DiT model's own response
to concept prompts — no pre-trained LoRAs required.

For each prompt the script runs a forward+backward pass through the base model
with random noise latents.  The weight gradient at each linear layer measures
"which directions in weight space does this conditioning activate?".  SVD on
the accumulated gradients gives the principal subspace for that concept type.

Tag-swap / delta mode (recommended):
    Supply a --neutral prompt (e.g. "a portrait").  For each forward pass the
    same noise and timestep is used for both the concept prompt and the neutral
    prompt.  The difference gradient (concept - neutral) cancels content noise
    and isolates concept-specific weight directions.

Output format is identical to extract_subspace.py — the SubspaceGuard and
restrict_to_mapped machinery consume it without any changes.

Usage
-----
    # Style subspace from artist tags, with tag-swap delta:
    python tools/extract_subspace_from_model.py \\
        --dit_path    X:/models/anima.safetensors \\
        --qwen3_path  X:/models/qwen3 \\
        --prompts     tools/prompts/style_artists.txt \\
        --neutral     "portrait of a person" \\
        --output      subspaces/style.safetensors \\
        --n_components 16 --plot_decay

    # Subject subspace, no neutral:
    python tools/extract_subspace_from_model.py \\
        --dit_path    X:/models/anima.safetensors \\
        --qwen3_path  X:/models/qwen3 \\
        --prompts     tools/prompts/subject_person.txt \\
        --output      subspaces/subject.safetensors

    # Restrict timestep range to match training preset:
    ... --min_timestep 50 --max_timestep 450
"""

import argparse
import gc
import os
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — mirrors train.py
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

# extract_subspace lives in the same directory — import directly
sys.path.insert(0, _TOOL_DIR)
from extract_subspace import plot_singular_value_decay, save_subspace

CUDA = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Minimal forward-pass context (anima_forward expects t.unet)
# ---------------------------------------------------------------------------

class _FwdCtx:
    """Minimal namespace satisfying the anima_forward(t, ...) interface."""
    def __init__(self, unet, text_model):
        self.unet = unet
        self.text_model = text_model


# ---------------------------------------------------------------------------
# Layer map: lora_name → linear module
# Mirrors the key-naming in LoRANetwork.create_modules so produced subspace
# keys match exactly what the training guard expects.
# ---------------------------------------------------------------------------

def build_target_layer_map(dit, module_filter: str = "!adaln_modulation") -> dict:
    """Return {lora_name: nn.Linear} for all LoRA-targetable layers in the DiT.

    The lora_name format is identical to what LoRANetwork.create_modules
    produces, ensuring full compatibility with the subspace guard.
    """
    layer_map: dict = {}
    for name, module in dit.named_modules():
        if module.__class__.__name__ not in ANIMA_TARGET_REPLACE_MODULE:
            continue
        for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ not in LORA_LINEAR:
                continue
            lora_name = (LORA_PREFIX_UNET + "." + name + "." + child_name).replace(".", "_")
            if module_filter.strip() and not _matches_module_filter(lora_name, module_filter):
                continue
            layer_map[lora_name] = child_module
    return layer_map


# ---------------------------------------------------------------------------
# Single-pass gradient collection
# ---------------------------------------------------------------------------

def _grads_for_pass(
    fwd_ctx: _FwdCtx,
    scheduler: AnimaFlowScheduler,
    layer_map: dict,
    cond,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    dtype: torch.dtype,
) -> dict:
    """Forward+backward for one (cond, noise, timestep) triple.

    Returns {lora_name: grad_tensor (float32)} and leaves all target weights
    with requires_grad=False and grad=None afterwards.
    """
    latents = torch.zeros_like(noise)
    noisy = scheduler.add_noise(latents, noise, timesteps).to(dtype)

    for module in layer_map.values():
        module.weight.requires_grad_(True)
        if module.weight.grad is not None:
            module.weight.grad = None

    with torch.enable_grad():
        # Autocast mirrors t.a.autocast() in train.py — required so that
        # float32 timestep embeddings are cast to match bfloat16 weights.
        with torch.autocast("cuda", dtype=dtype):
            pred = anima_forward(fwd_ctx, noisy, timesteps, cond)
        # Loss in float32 outside autocast for gradient numerical stability.
        # Velocity target = noise − latents = noise (latents are zero).
        # Loss magnitude is irrelevant — only gradient *direction* matters.
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()

    grads: dict = {}
    for lora_name, module in layer_map.items():
        if module.weight.grad is not None:
            grads[lora_name] = module.weight.grad.detach().float().clone()
        module.weight.requires_grad_(False)
        module.weight.grad = None

    return grads


# ---------------------------------------------------------------------------
# Multi-prompt gradient accumulation
# ---------------------------------------------------------------------------

def collect_gradients(
    fwd_ctx: _FwdCtx,
    scheduler: AnimaFlowScheduler,
    layer_map: dict,
    concept_prompts: list,
    neutral_cond,
    text_model: AnimaTextModel,
    args,
) -> dict:
    """Accumulate per-layer gram matrices (G^T G) entirely on GPU.

    For each (prompt, sample) pair:
      - Draws fixed-seed noise so concept and neutral see identical latents.
      - If neutral_cond is supplied, subtracts neutral gradient from concept
        gradient (tag-swap delta) to cancel content noise.
      - Immediately folds each gradient into gram[layer] += g.T @ g on the GPU.
        The raw gradient tensor is discarded; only the (in×in) gram matrix is
        kept between passes.

    Returns {lora_name: Tensor(in, in) float32 on CUDA}.

    Memory: ~1–2 GB for gram matrices (258 layers at typical Anima dimensions),
    well within a 96 GB VRAM budget.  No CPU transfers or disk I/O during
    collection — the matmul and accumulation happen entirely on-device.
    """
    dtype = _parse_dtype(args.precision)
    gram: dict = {}  # lora_name -> Tensor(in, in) float32, on CUDA

    total = len(concept_prompts)
    print(
        f"\nCollecting gradients — {total} prompts × {args.samples_per_prompt} samples "
        f"= {total * args.samples_per_prompt} passes"
        + (" (with tag-swap delta)" if neutral_cond is not None else "")
    )
    print(f"Timestep range: [{args.min_timestep}, {args.max_timestep}]")

    for p_idx, prompt in enumerate(concept_prompts):
        concept_cond, _ = text_model.encode_text([prompt])
        concept_cond = move_cond_to_device(concept_cond, CUDA, dtype)

        for s_idx in range(args.samples_per_prompt):
            # Deterministic seed per (prompt, sample) → same noise for neutral
            seed = p_idx * 10000 + s_idx
            gen = torch.Generator(device=CUDA).manual_seed(seed)
            noise = torch.randn(
                1, args.latent_channels, args.latent_height, args.latent_width,
                device=CUDA, dtype=dtype, generator=gen,
            )
            timesteps = torch.randint(
                args.min_timestep, args.max_timestep, (1,),
                device=CUDA, generator=torch.Generator(device=CUDA).manual_seed(seed),
            )

            concept_grads = _grads_for_pass(
                fwd_ctx, scheduler, layer_map, concept_cond, noise, timesteps, dtype
            )

            if neutral_cond is not None:
                neutral_grads = _grads_for_pass(
                    fwd_ctx, scheduler, layer_map,
                    move_cond_to_device(neutral_cond, CUDA, dtype),
                    noise, timesteps, dtype,
                )
                for key in concept_grads:
                    if key in neutral_grads:
                        concept_grads[key] = concept_grads[key] - neutral_grads[key]

            # Fold gradient into gram matrix on GPU — raw gradient is discarded.
            for key, g in concept_grads.items():
                contrib = g.t() @ g  # (in, in), stays on CUDA
                if key in gram:
                    gram[key] += contrib
                else:
                    gram[key] = contrib

            del concept_grads

        print(f"  [{p_idx + 1:>{len(str(total))}}/{total}] {prompt[:72]}")

    print()
    return gram


# ---------------------------------------------------------------------------
# SVD  (identical logic to extract_subspace.py)
# ---------------------------------------------------------------------------

def subspace_from_gradients(gram_per_layer: dict, n_components: int) -> dict:
    """Eigendecompose per-layer gram matrices. Output format identical to extract_subspace.py.

    The eigenvectors of G^T G are the right singular vectors of the stacked
    gradient matrix, so the result is equivalent to SVD on accumulated raw
    gradients.  eigh runs on the same device as the gram matrix (CUDA), and the
    result is only moved to CPU as fp16 at save time.

    Eigenvalues from torch.linalg.eigh are in ascending order; we reverse so
    that V_K[:,0] is the most important direction (largest singular value).
    """
    subspaces: dict = {}
    skipped = 0

    for lora_name, gram in sorted(gram_per_layer.items()):
        n = gram.shape[0]
        K = min(n_components, n - 1)

        if K < 1:
            skipped += 1
            continue

        try:
            # eigh runs on CUDA — no device transfer until final .cpu() below
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            # Flip to descending order; take top-K
            V_K = eigenvectors[:, -K:].flip(dims=[1]).contiguous()  # (n, K)
            # Eigenvalue = singular_value²; clamp for numerical safety
            S = eigenvalues[-K:].flip(dims=[0]).clamp(min=0).sqrt()
            subspaces[lora_name] = {
                "V_K": V_K.half().cpu(),
                "singular_values": S.half().cpu(),
            }
        except Exception as exc:
            print(f"  Warning: eigh failed for {lora_name}: {exc}")
            skipped += 1

    print(
        f"Extracted subspaces for {len(subspaces)}/{len(gram_per_layer)} layers"
        + (f" ({skipped} skipped)." if skipped else ".")
    )
    return subspaces


# ---------------------------------------------------------------------------
# Model loading  (mirrors train_main in train.py)
# ---------------------------------------------------------------------------

def load_models(args):
    from library import anima_utils

    dtype = _parse_dtype(args.precision)

    print(f"Loading Qwen3 text encoder from {args.qwen3_path} ...")
    qwen3_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3_path, dtype=dtype, device="cpu"
    )
    qwen3_encoder = qwen3_encoder.to(CUDA)
    qwen3_encoder.requires_grad_(False)
    qwen3_encoder.eval()

    text_model = AnimaTextModel(
        qwen3_encoder,
        args.qwen3_path,
        args.t5_tokenizer_path or None,
        device=CUDA,
        dtype=dtype,
    )

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

    # VideoRopePosition3DEmb registers `seq`, `dim_spatial_range`, and
    # `dim_temporal_range` as buffers, but they are (re)initialised on CPU
    # during __init__ / reset_parameters before the weights are loaded.
    # If load_anima_model only maps weight tensors, these buffers stay on
    # CPU and cause a device mismatch inside _apply_rotary_pos_emb_base.
    # Explicitly moving the embedder(s) to CUDA fixes this.
    for attr in ("pos_embedder", "extra_pos_embedder"):
        emb = getattr(dit, attr, None)
        if emb is not None:
            emb.to(CUDA)

    return dit, text_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_prompts(path: str) -> list:
    """Load prompts from a text file. Blank lines and # comments are skipped."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract principal subspace from Anima DiT model response to prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model paths
    parser.add_argument("--dit_path",          required=True, help="Path to Anima DiT weights (.safetensors)")
    parser.add_argument("--qwen3_path",         required=True, help="Path to Qwen3 text encoder directory")
    parser.add_argument("--t5_tokenizer_path",  default="",    help="Path to T5 tokenizer directory (optional)")

    # Prompts
    parser.add_argument("--prompts",  required=True, help="Text file of concept prompts (one per line, # = comment)")
    parser.add_argument("--neutral",  default="",
                        help="Neutral prompt for tag-swap delta (e.g. 'portrait of a person'). "
                             "Omit to use raw gradients without subtraction.")

    # Output
    parser.add_argument("--output",       required=True, help="Output subspace .safetensors file path")
    parser.add_argument("--n_components", type=int, default=16,
                        help="Principal components per layer (default: 16). "
                             "Use --plot_decay to inspect the decay curve and tune this.")

    # Sampling
    parser.add_argument("--samples_per_prompt", type=int, default=4,
                        help="Forward+backward passes per prompt (default: 4). "
                             "More = richer subspace estimate, slower.")
    parser.add_argument("--min_timestep", type=int, default=0,
                        help="Minimum timestep to sample (default: 0). "
                             "Match to your training preset's train_min_timesteps.")
    parser.add_argument("--max_timestep", type=int, default=1000,
                        help="Maximum timestep to sample (default: 1000). "
                             "Match to your training preset's train_max_timesteps.")

    # Latent shape — no VAE needed, we use random noise
    parser.add_argument("--latent_channels", type=int, default=16,
                        help="Latent channel count (default: 16)")
    parser.add_argument("--latent_height",   type=int, default=32,
                        help="Latent spatial height (default: 32 → 512px with 16× VAE)")
    parser.add_argument("--latent_width",    type=int, default=32,
                        help="Latent spatial width (default: 32 → 512px with 16× VAE)")

    # Layer filter
    parser.add_argument("--module_filter", default="!adaln_modulation",
                        help="Layer regex filter, same syntax as network_module_filter "
                             "(default: '!adaln_modulation')")

    # Precision / misc
    parser.add_argument("--precision", default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model precision (default: bfloat16)")
    parser.add_argument("--plot_decay", action="store_true",
                        help="Save a singular value decay plot alongside the output (requires matplotlib)")

    args = parser.parse_args()

    if args.min_timestep >= args.max_timestep:
        parser.error("--min_timestep must be less than --max_timestep")

    if not os.path.isfile(args.prompts):
        parser.error(f"Prompts file not found: {args.prompts}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    dit, text_model = load_models(args)
    scheduler = AnimaFlowScheduler()
    fwd_ctx = _FwdCtx(dit, text_model)

    # ------------------------------------------------------------------
    # Build layer map
    # ------------------------------------------------------------------
    layer_map = build_target_layer_map(dit, module_filter=args.module_filter)
    print(f"Targeting {len(layer_map)} linear layers (filter: '{args.module_filter}').")

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    concept_prompts = load_prompts(args.prompts)
    if not concept_prompts:
        print("Error: no prompts found in file (check for blank lines or # comments).")
        sys.exit(1)
    print(f"Loaded {len(concept_prompts)} concept prompts from {args.prompts}")

    # ------------------------------------------------------------------
    # Optional neutral conditioning for tag-swap delta
    # ------------------------------------------------------------------
    neutral_cond = None
    if args.neutral.strip():
        dtype = _parse_dtype(args.precision)
        neutral_cond, _ = text_model.encode_text([args.neutral.strip()])
        neutral_cond = move_cond_to_device(neutral_cond, CUDA, dtype)
        print(f"Neutral prompt: \"{args.neutral}\"")
    else:
        print("No neutral prompt — using raw gradients (no tag-swap delta).")

    # ------------------------------------------------------------------
    # Collect gradients → gram matrices (on GPU)
    # ------------------------------------------------------------------
    gram = collect_gradients(
        fwd_ctx, scheduler, layer_map,
        concept_prompts, neutral_cond, text_model, args,
    )

    # Models no longer needed — free VRAM before eigh.
    del dit, fwd_ctx
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Eigendecompose → save
    # ------------------------------------------------------------------
    subspaces = subspace_from_gradients(gram, args.n_components)
    if not subspaces:
        print("Error: no subspaces extracted.")
        sys.exit(1)

    save_subspace(subspaces, args.output)

    if args.plot_decay:
        plot_singular_value_decay(subspaces, args.output)

    print(f"\nDone. Set subspace_guard_path to:\n  {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
