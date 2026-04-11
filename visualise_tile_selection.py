"""
Visualise texture-mode tile selection over a folder of dataset images.

Shows N iterations of the JIT crop-selection loop (up to 10 attempts each),
colour-coded by outcome:
  - Green  : accepted early (meets both energy + mask thresholds)
  - Yellow : best-so-far selected after exhausting attempts
  - Red    : rejected / not chosen (any attempt that was beaten or failed)

One PNG is written per image to the output folder, named <stem>_tiles.png.
Masks are auto-detected alongside each image as <stem>_mask.<ext> or
<stem>.mask.<ext>, or supplied via --mask_dir.

Usage
-----
python visualise_tile_selection.py \
    --input_dir  path/to/images/ \
    --output_dir path/to/out/ \
    [--mask_dir  path/to/masks/] \
    [--iterations 20] \
    [--tile_size 0] \
    [--energy_threshold 0.0] \
    [--metric laplacian|coherence|saturation|edge_count|combined] \
    [--no_avoid_masked] \
    [--canvas 1024x1024] \
    [--seed 42]

Metrics
-------
  laplacian  – Variance of the 3×3 Laplacian (classic sharpness proxy).
               Fast but fooled by film grain and sensor noise.
  coherence  – Gradient-coherence weighted by magnitude (structure tensor).
               High for organised subject edges; near-zero for noise/grain
               and featureless areas.  Best single "find the character" metric.
  saturation – Mean HSV saturation.  Subjects tend to be more colourful than
               blurry or noisy neutral backgrounds.
  edge_count – Fraction of pixels whose gradient magnitude exceeds a fixed
               threshold.  Rewards tiles with real structural edges rather
               than many weak noise responses.
  combined   – coherence × (1 + 2·saturation).  Rewards colourful, structured
               subjects while doubly penalising noise and flat backgrounds.
"""

import argparse
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

METRIC_NAMES = ("laplacian", "coherence", "saturation", "edge_count", "combined")

# ──────────────────────────────────────────────────────────────────────────────
# Core logic mirrored from dataset.py  (no torch / training dependencies)
# ──────────────────────────────────────────────────────────────────────────────

_TEXTURE_CANVAS_PRESETS = [
    (640, 1536), (1536, 640),
    (832, 1216), (1216, 832),
    (1024, 1024),
]


def _box_smooth(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Uniform box filter with reflect-padding; output same shape as input."""
    pad = k // 2
    a_p = np.pad(a, pad, mode="reflect")
    cs = a_p.cumsum(axis=0)
    out = cs[k:] - cs[:-k]
    cs2 = out.cumsum(axis=1)
    return (cs2[:, k:] - cs2[:, :-k]) / (k * k)


def _compute_metrics(pil_crop: Image.Image, mask_crop: "Image.Image | None") -> dict:
    """
    Compute all crop quality metrics for character/subject detection.

    Returns a dict: metric_name → float (all non-negative).

    laplacian  – Laplacian variance: classic sharpness, but fooled by grain.
    coherence  – Structure-tensor coherence weighted by gradient magnitude.
                 High for organised subject edges, near-zero for noise or
                 flat regions.  Best discriminator for real structure vs grain.
    saturation – Mean HSV saturation.  Subjects tend to be more colourful
                 than out-of-focus or desaturated backgrounds.
    edge_count – Fraction of pixels with gradient magnitude > 0.05 (on [0,1]
                 image).  Rewards real structural edges over diffuse noise.
    combined   – coherence × (1 + 2·saturation): rewards colourful, structured
                 subjects; doubly penalises noise and featureless backgrounds.
    """
    rgb  = np.array(pil_crop.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Mask: same spatial size as crop, float32 in [0, 1]
    mask_np: "np.ndarray | None" = None
    if mask_crop is not None:
        m = mask_crop.convert("L")
        if m.size != pil_crop.size:
            m = m.resize(pil_crop.size, Image.BILINEAR)
        mask_np = np.array(m, dtype=np.float32) / 255.0

    eps = 1e-8

    # ── 1. Laplacian energy ───────────────────────────────────────────────────
    lap = (gray[1:-1, 1:-1] * 4
           - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    if mask_np is not None:
        lap = lap * mask_np[1:-1, 1:-1]
    laplacian = float(np.var(lap))

    # ── 2. Gradient coherence (structure tensor) ──────────────────────────────
    # Ix, Iy via central differences (np.gradient); smooth structure tensor
    # components with a local box filter to aggregate neighbourhood evidence.
    Iy, Ix = np.gradient(gray)
    Ixx = _box_smooth(Ix * Ix)
    Iyy = _box_smooth(Iy * Iy)
    Ixy = _box_smooth(Ix * Iy)
    trace = Ixx + Iyy                       # λ1 + λ2  (gradient energy proxy)
    det   = Ixx * Iyy - Ixy * Ixy          # λ1 · λ2
    # coherence = (λ1−λ2)²/(λ1+λ2)²  ∈ [0,1]:
    #   → 1 when one dominant direction (real edge); → 0 for isotropic noise
    coh = np.where(
        trace > eps,
        np.clip(trace**2 - 4.0 * det, 0.0, None) / (trace**2 + eps),
        0.0,
    )
    weighted_coh = coh * trace              # weight by local gradient energy
    if mask_np is not None:
        weighted_coh = weighted_coh * mask_np
    coherence = float(np.mean(weighted_coh))

    # ── 3. Saturation ─────────────────────────────────────────────────────────
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    sat  = np.where(cmax > eps, (cmax - cmin) / cmax, 0.0)
    if mask_np is not None:
        sat = sat * mask_np
    saturation = float(sat.mean())

    # ── 4. Edge-count score ───────────────────────────────────────────────────
    # Fraction of pixels with a strong gradient.  Noise scatters many weak
    # responses; real subject edges produce fewer but stronger ones.
    EDGE_THRESH = 0.05           # on a [0,1]-normalised image
    grad_mag   = np.sqrt(Ix**2 + Iy**2)
    strong     = (grad_mag > EDGE_THRESH).astype(np.float32)
    if mask_np is not None:
        strong = strong * mask_np
        denom  = float(mask_np.sum()) + eps
    else:
        denom  = float(gray.size)
    edge_count = float(strong.sum() / denom)

    # ── 5. Combined ───────────────────────────────────────────────────────────
    combined = coherence * (1.0 + 2.0 * saturation)

    return {
        "laplacian":  laplacian,
        "coherence":  coherence,
        "saturation": saturation,
        "edge_count": edge_count,
        "combined":   combined,
    }


def _mask_score(mask_np_full: "np.ndarray | None", src_x, src_y, src_px) -> float:
    if mask_np_full is None:
        return 1.0
    region = mask_np_full[src_y:src_y + src_px, src_x:src_x + src_px]
    return float(region.mean())


def simulate_iteration(
    image: Image.Image,
    mask: "Image.Image | None",
    mask_np_full: "np.ndarray | None",
    canvas_hw: tuple,
    tile_size: int,
    energy_threshold: float,
    metric: str = "laplacian",
    max_attempts: int = 10,
) -> dict:
    """
    Run one dataset __getitem__ JIT-crop selection.

    If mask_np_full is provided, mask avoidance is always active:
      - mask_score = fraction of fully-opaque (unmasked) pixels in the crop
      - ms == 1.0  → zero masked pixels → eligible for early accept
      - ms <  1.0  → some masked pixels → keep trying, track best
    mask_score is the PRIMARY sort key; the selected metric (if enabled) is a
    secondary tiebreak among crops with equal mask scores.

    The caller controls whether mask avoidance is active by passing
    mask_np_full=None to disable it regardless of mask image presence.
    """
    canvas_h, canvas_w = canvas_hw
    clat_h, clat_w = canvas_h // 8, canvas_w // 8

    has_mask   = mask_np_full is not None
    use_metric = energy_threshold > 0

    effective_max = max_attempts if (has_mask or use_metric) else 1

    best_crop       = None
    best_ms         = -1.0
    best_metric_val = -1.0
    accepted_early  = False
    attempts        = []

    for attempt in range(effective_max):
        # ── Resolve source crop size in pixels ───────────────────────────────
        if tile_size > 0:
            src_px = min(tile_size, image.width, image.height)
        else:
            max_lat = min(image.width // 8, image.height // 8, clat_h, clat_w)
            min_lat = max(1, max_lat // 4)
            src_px  = random.randint(min_lat, max_lat) * 8
            src_px  = min(src_px, image.width, image.height)

        src_y = random.randint(0, image.height - src_px)
        src_x = random.randint(0, image.width - src_px)

        candidate_crop = image.crop((src_x, src_y, src_x + src_px, src_y + src_px))

        # Mask score: 1.0 = fully unmasked, <1.0 = some masked pixels present
        ms = _mask_score(mask_np_full, src_x, src_y, src_px) if has_mask else 1.0

        # Quality metrics — always compute all; mask crop supplied when available
        mask_crop = (
            mask.crop((src_x, src_y, src_x + src_px, src_y + src_px))
            if mask is not None else None
        )
        metrics = _compute_metrics(candidate_crop, mask_crop if has_mask else None)
        active_metric_val = metrics[metric] if use_metric else None

        attempt_rec = {
            "attempt":    attempt,
            "src_x": src_x, "src_y": src_y, "src_px": src_px,
            "mask_score": ms,
            "metrics":    metrics,
            "status":     "rejected",
        }

        # Update best: mask coverage PRIMARY, active metric secondary tiebreak
        mv = active_metric_val if active_metric_val is not None else -1.0
        if ms > best_ms or (ms == best_ms and mv > best_metric_val):
            best_ms         = ms
            best_metric_val = mv
            best_crop       = attempt_rec

        attempts.append(attempt_rec)

        # Criteria met — accept and stop.  Breaking here means no subsequent
        # crops are sampled, so nothing valid can be falsely coloured red.
        if ms >= 1.0 and ((not use_metric) or active_metric_val >= energy_threshold):
            attempt_rec["status"] = "accepted_early"
            accepted_early = True
            break

    # Exhausted all attempts without meeting criteria — take the best overall
    if not accepted_early and best_crop is not None:
        best_crop["status"] = "best"

    return {"attempts": attempts, "best": best_crop, "canvas_hw": canvas_hw}


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

STATUS_COLOR = {
    "accepted_early": "#00cc44",   # green
    "best":           "#ffaa00",   # amber
    "rejected":       "#cc2200",   # red
}
STATUS_LABEL = {
    "accepted_early": "Accepted (early)",
    "best":           "Best (fallback)",
    "rejected":       "Rejected",
}


def _build_intersection_overlay(records, mask_np_full, image_shape):
    """Return a single RGBA float32 array (red) showing every crop×mask
    intersection across the given records, or None if no intersections exist."""
    if mask_np_full is None:
        return None
    h, w = image_shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    for rec_group in records:
        for rec in rec_group["attempts"]:
            if rec["mask_score"] >= 1.0:
                continue
            x, y, s = rec["src_x"], rec["src_y"], rec["src_px"]
            region = mask_np_full[y:y + s, x:x + s]
            np.maximum(acc[y:y + s, x:x + s], 1.0 - region, out=acc[y:y + s, x:x + s])
    if acc.max() == 0:
        return None
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0          # red
    overlay[..., 2] = 1.0          # blue
    overlay[..., 3] = acc * 0.55
    return overlay


def _metric_label(rec: dict, active_metric: "str | None") -> str:
    """Format a compact metric label for a key crop rectangle."""
    m = rec["mask_score"]
    mets = rec.get("metrics")
    if mets is None:
        return f"m={m:.2f}"
    # Short names to keep labels small
    short = {"laplacian": "lap", "coherence": "coh", "saturation": "sat",
             "edge_count": "edg", "combined": "comb"}
    # Active metric first (if any), then the rest in a fixed order
    order = list(METRIC_NAMES)
    if active_metric and active_metric in order:
        order.remove(active_metric)
        order.insert(0, active_metric)
    parts = [f"{short.get(k, k)}={mets[k]:.3f}" for k in order if k in mets]
    # Two metrics per line, mask score at end
    lines = []
    for i in range(0, len(parts), 2):
        lines.append("  ".join(parts[i:i + 2]))
    lines.append(f"m={m:.2f}")
    return "\n".join(lines)


def _draw_rects(ax, records, active_metric: "str | None", labels=True):
    """Draw attempt rectangles from one or more iteration records onto ax."""
    for iteration_rec in records:
        for rec in iteration_rec["attempts"]:
            color  = STATUS_COLOR[rec["status"]]
            is_key = rec["status"] != "rejected"
            lw     = 2.5 if is_key else 1.0

            rect = mpatches.Rectangle(
                (rec["src_x"], rec["src_y"]), rec["src_px"], rec["src_px"],
                linewidth=lw, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

            if labels and is_key:
                label = _metric_label(rec, active_metric)
                ax.text(
                    rec["src_x"] + 2, rec["src_y"] + 12,
                    label,
                    color=color, fontsize=5.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="square,pad=0.1", fc="black", alpha=0.5, lw=0),
                )


def _draw_iteration_panel(ax, image_rgba, mask_overlay, mask_np_full, iteration_rec, idx, active_metric):
    """Draw one iteration panel: image + all attempted rectangles."""
    ax.imshow(image_rgba)
    if mask_overlay is not None:
        ax.imshow(mask_overlay, alpha=0.3)
    isect = _build_intersection_overlay([iteration_rec], mask_np_full, image_rgba.shape)
    if isect is not None:
        ax.imshow(isect)
    _draw_rects(ax, [iteration_rec], active_metric, labels=True)
    canvas_h, canvas_w = iteration_rec["canvas_hw"]
    ax.set_title(f"Iter {idx}  canvas={canvas_w}×{canvas_h}", fontsize=7)
    ax.axis("off")


def visualise(
    image_path: str,
    mask_path: "str | None",
    n_iterations: int,
    tile_size: int,
    energy_threshold: float,
    metric: str,
    avoid_masked: bool,
    canvas_str: "str | None",
    output_path: str,
    seed: "int | None" = None,
    split: bool = False,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    image_pil = Image.open(image_path)

    # Mirror dataset.py: use the image's own alpha channel as the mask when
    # the file is RGBA and no external mask file is provided.
    mask_l: "Image.Image | None" = None
    if image_pil.mode == "RGBA":
        mask_l = image_pil.split()[3]          # already "L" mode

    # External mask file overrides image alpha (same priority as dataset.py's
    # texture_mask_directory).  For RGBA mask files, use the alpha channel
    # rather than luminance so that white-on-transparent masks work correctly.
    if mask_path:
        ext = Image.open(mask_path)
        mask_l = ext.split()[3] if ext.mode == "RGBA" else ext.convert("L")

    image = image_pil.convert("RGB")
    # Keep a PIL mask for crop-level operations inside simulate_iteration
    mask  = mask_l

    mask_np_full = None
    mask_overlay = None
    if mask_l is not None:
        mask_np_full = np.array(mask_l, dtype=np.float32) / 255.0
        # Red overlay on masked (dark/bad) pixels, not on the good clear areas
        red = np.zeros((*mask_np_full.shape, 4), dtype=np.uint8)
        red[..., 0] = 220
        red[..., 3] = ((1.0 - mask_np_full) * 200).astype(np.uint8)
        mask_overlay = red

    image_rgba = np.array(image)

    # Canvas selection
    if canvas_str:
        w, h = (int(v) for v in canvas_str.lower().split("x"))
        fixed_canvas = (h, w)
    else:
        fixed_canvas = None

    # ── Run simulations ────────────────────────────────────────────────────────
    records = []
    for _ in range(n_iterations):
        canvas_hw = fixed_canvas if fixed_canvas else random.choice(_TEXTURE_CANVAS_PRESETS)
        # Pass mask_np_full=None when avoidance is disabled so the inner
        # loop never scores or retries on mask coverage.
        active_mask_np = mask_np_full if avoid_masked else None
        rec = simulate_iteration(
            image, mask, active_mask_np, canvas_hw,
            tile_size, energy_threshold, metric,
        )
        records.append(rec)

    active_metric = metric if energy_threshold > 0 else None

    legend_handles = [
        mpatches.Patch(color=STATUS_COLOR["accepted_early"], label=STATUS_LABEL["accepted_early"]),
        mpatches.Patch(color=STATUS_COLOR["best"],           label=STATUS_LABEL["best"]),
        mpatches.Patch(color=STATUS_COLOR["rejected"],       label=STATUS_LABEL["rejected"]),
    ]

    title_parts = [f"Tile selection — {os.path.basename(image_path)}"]
    if energy_threshold > 0:
        title_parts.append(f"{metric}>={energy_threshold:.4f}")
    if tile_size:
        title_parts.append(f"tile_size={tile_size}px")
    if avoid_masked and mask_l is not None:
        title_parts.append("avoid_masked=True")
    title = "  |  ".join(title_parts)

    if split:
        # ── Grid: one panel per iteration ─────────────────────────────────────
        cols = min(5, n_iterations)
        rows = math.ceil(n_iterations / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
        axes_flat = np.array(axes).flatten() if n_iterations > 1 else [axes]

        for idx, (ax, rec) in enumerate(zip(axes_flat, records)):
            _draw_iteration_panel(ax, image_rgba, mask_overlay, mask_np_full, rec, idx, active_metric)
        for ax in axes_flat[n_iterations:]:
            ax.axis("off")

        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
                   framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(title, fontsize=8, y=1.01)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        # ── Overlay: all iterations on one image ──────────────────────────────
        ih, iw = image_rgba.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(max(6, iw / 100), max(5, ih / 100)))
        ax.imshow(image_rgba)
        if mask_overlay is not None:
            ax.imshow(mask_overlay, alpha=0.3)
        isect = _build_intersection_overlay(records, mask_np_full, image_rgba.shape)
        if isect is not None:
            ax.imshow(isect)
        _draw_rects(ax, records, active_metric, labels=False)
        _draw_rects(ax, records, active_metric, labels=True)
        ax.axis("off")
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
                   framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
        n_attempts = sum(len(r["attempts"]) for r in records)
        fig.suptitle(f"{title}  |  {n_iterations} iters  {n_attempts} attempts", fontsize=8, y=1.01)
        plt.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")

    # ── Print summary stats ───────────────────────────────────────────────────
    n_early  = sum(1 for r in records if r["best"] and r["best"]["status"] == "accepted_early")
    n_best   = sum(1 for r in records if r["best"] and r["best"]["status"] == "best")
    n_total_attempts = sum(len(r["attempts"]) for r in records)
    print(f"\n  Iterations       : {n_iterations}")
    print(f"  Accepted early   : {n_early}  ({100*n_early/n_iterations:.0f}%)")
    print(f"  Fallback best    : {n_best}  ({100*n_best/n_iterations:.0f}%)")
    print(f"  Avg attempts/iter: {n_total_attempts/n_iterations:.1f}")
    if energy_threshold > 0:
        vals = [
            r["best"]["metrics"][metric]
            for r in records
            if r["best"] and r["best"].get("metrics")
        ]
        if vals:
            print(f"  {metric} (selected): min={min(vals):.5f}  max={max(vals):.5f}  mean={np.mean(vals):.5f}")
    # Always print a summary of all metrics for the selected crops
    all_mets = [r["best"]["metrics"] for r in records if r["best"] and r["best"].get("metrics")]
    if all_mets:
        print(f"\n  Metric summary (selected crops):")
        for k in METRIC_NAMES:
            vs = [m[k] for m in all_mets]
            print(f"    {k:12s}: min={min(vs):.5f}  max={max(vs):.5f}  mean={np.mean(vs):.5f}")


# ──────────────────────────────────────────────────────────────────────────────

def _find_mask(image_path: Path, mask_dir: "Path | None") -> "Path | None":
    """Look for a mask file paired with an image."""
    stem = image_path.stem
    # Explicit mask dir: look for same stem with any image extension
    if mask_dir is not None:
        for ext in IMAGE_EXTS:
            candidate = mask_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None
    # Alongside the image: <stem>_mask.<ext> or <stem>.mask.<ext>
    for suffix in (f"_mask", f".mask"):
        for ext in IMAGE_EXTS:
            candidate = image_path.parent / (stem + suffix + ext)
            if candidate.exists():
                return candidate
    return None


def _process_one(kwargs: dict) -> str:
    """Worker entry-point (must be module-level for pickle on Windows spawn)."""
    visualise(**kwargs)
    return kwargs["image_path"]


def process_folder(
    input_dir: str,
    output_dir: str,
    mask_dir: "str | None",
    n_iterations: int,
    tile_size: int,
    energy_threshold: float,
    metric: str,
    avoid_masked: bool,
    canvas_str: "str | None",
    seed: "int | None",
    workers: int = 0,
    split: bool = False,
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    mask_path   = Path(mask_dir) if mask_dir else None

    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {input_dir}")
        return

    n_workers = workers if workers > 0 else os.cpu_count() or 1
    print(f"Found {len(images)} image(s) in {input_dir}  |  workers={n_workers}")

    jobs = []
    for i, img_file in enumerate(images):
        found_mask = _find_mask(img_file, mask_path)
        jobs.append(dict(
            image_path       = str(img_file),
            mask_path        = str(found_mask) if found_mask else None,
            n_iterations     = n_iterations,
            tile_size        = tile_size,
            energy_threshold = energy_threshold,
            metric           = metric,
            avoid_masked     = avoid_masked,
            canvas_str       = canvas_str,
            output_path      = str(output_path / (img_file.stem + "_tiles.png")),
            seed             = (seed + i) if seed is not None else None,
            split            = split,
        ))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, j): j["image_path"] for j in jobs}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"ERROR {futures[fut]}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir",  required=True, help="Folder containing dataset images")
    parser.add_argument("--output_dir", help="Folder to write visualisation PNGs")
    parser.add_argument("--mask_dir",   default=None,  help="Folder with mask images (matched by stem); "
                                                             "if omitted, masks are auto-detected alongside each image")
    parser.add_argument("--iterations", type=int,   default=20,  help="Dataset iterations per image (default: 20)")
    parser.add_argument("--tile_size",  type=int,   default=0,   help="Source crop square side in pixels (0=random)")
    parser.add_argument("--energy_threshold", type=float, default=0.0,
                        help="Quality threshold for the selected metric (0=disabled)")
    parser.add_argument("--metric", default="laplacian", choices=list(METRIC_NAMES),
                        help="Quality metric used for thresholding and crop selection "
                             "(default: laplacian).  All metrics are always computed and "
                             "shown in labels regardless of this choice.")
    parser.add_argument("--no_avoid_masked", action="store_true", help="Disable mask-avoidance logic")
    parser.add_argument("--canvas",  default=None, help="Fixed canvas WxH e.g. 1024x1024 (default: random preset)")
    parser.add_argument("--seed",    type=int, default=None, help="RNG seed (offset per image for variety)")
    parser.add_argument("--workers", type=int, default=0,    help="Parallel worker processes (0 = cpu_count)")
    parser.add_argument("--split",   action="store_true",   help="One panel per iteration instead of a single overlay image")
    args = parser.parse_args()

    if args.output_dir == None:
        args.output_dir = Path(args.input_dir) / "visu"

    process_folder(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        mask_dir         = args.mask_dir,
        n_iterations     = args.iterations,
        tile_size        = args.tile_size,
        energy_threshold = args.energy_threshold,
        metric           = args.metric,
        avoid_masked     = not args.no_avoid_masked,
        canvas_str       = args.canvas,
        seed             = args.seed,
        workers          = args.workers,
        split            = args.split,
    )


if __name__ == "__main__":
    main()
