"""
extract_texture_crops.py — Texture-focused crop extractor for style training

Scans a directory of training images and extracts high-texture regions as square
crops, preserving caption/text files alongside them. The crops capture fine
detail (brush strokes, linework, hatching) at near-native resolution, bypassing
the resolution loss that bucketing+resize introduces.

Usage:
    python extract_texture_crops.py --input_dir /path/to/images [options]

Output structure mirrors the input directory inside <output_dir>, e.g.:
    input:  training/artist/img001.png  →  training/artist/img001.txt
    output: training_texture/artist/img001_crop0.png  →  …_crop0.txt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Extensions matching trainer/dataset.py TARGET_IMAGEFILES
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp", ".pcx", ".ico"}


# ---------------------------------------------------------------------------
# Texture scoring
# ---------------------------------------------------------------------------

def compute_texture_score_map(gray: np.ndarray, analysis_radius: int = 16) -> np.ndarray:
    """
    Return a per-pixel texture richness map in [0, 1].

    Two complementary signals are blended:
      - Laplacian energy  : squared Laplacian response, captures fine detail,
                            thin lines, sharp edges, cross-hatching.
      - Gradient magnitude: Sobel magnitude, captures edge density and stroke
                            boundaries.

    Both are smoothed over a neighbourhood of radius `analysis_radius` so each
    pixel's score reflects the richness of its surrounding region.
    """
    gray_f = gray.astype(np.float32)

    # --- Laplacian energy ------------------------------------------------
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    lap_energy = lap ** 2

    # --- Gradient magnitude ----------------------------------------------
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # --- Smooth to get local means ---------------------------------------
    ksize = 2 * analysis_radius + 1
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)

    local_lap = cv2.filter2D(lap_energy, -1, kernel)
    local_grad = cv2.filter2D(grad_mag, -1, kernel)

    def safe_norm(x: np.ndarray) -> np.ndarray:
        mx = float(x.max())
        return x / mx if mx > 1e-8 else np.zeros_like(x)

    score = 0.5 * safe_norm(local_lap) + 0.5 * safe_norm(local_grad)
    return score  # shape (H, W), dtype float32


def score_all_crops(score_map: np.ndarray, crop_size: int, stride: int) -> list[tuple[float, int, int]]:
    """
    Efficiently score every candidate crop window using an integral-image-like
    approach (box filter on the score map, then read off positions).

    Returns a list of (score, x, y) sorted descending by score.
    """
    H, W = score_map.shape

    # Box filter with kernel = crop_size gives us the SUM inside each window.
    # We want the mean, but we only use scores for ranking, so the sum is fine.
    box_sum = cv2.boxFilter(score_map, -1, (crop_size, crop_size),
                            normalize=False, borderType=cv2.BORDER_CONSTANT)

    candidates = []
    for y in range(0, H - crop_size + 1, stride):
        for x in range(0, W - crop_size + 1, stride):
            # boxFilter result at (y + crop_size//2, x + crop_size//2) is the
            # sum over the crop. We use the top-left corner for simplicity and
            # read the centre pixel of the kernel response.
            cy = min(y + crop_size // 2, H - 1)
            cx = min(x + crop_size // 2, W - 1)
            s = float(box_sum[cy, cx])
            candidates.append((s, x, y))

    candidates.sort(reverse=True)
    return candidates


def greedy_nms(candidates: list[tuple[float, int, int]],
               crop_size: int,
               max_crops: int,
               min_score: float,
               overlap_threshold: float) -> list[tuple[float, int, int]]:
    """
    Greedy non-maximum suppression: iterates candidates from highest to lowest
    score, keeping a crop only if it doesn't overlap too much with any already
    selected crop.
    """
    selected: list[tuple[float, int, int]] = []

    for score, x, y in candidates:
        if score < min_score:
            break
        if len(selected) >= max_crops:
            break

        overlap = False
        for _, sx, sy in selected:
            # Intersection-over-union for two axis-aligned squares of equal size
            ix = max(0, min(x + crop_size, sx + crop_size) - max(x, sx))
            iy = max(0, min(y + crop_size, sy + crop_size) - max(y, sy))
            iou = (ix * iy) / (crop_size ** 2)
            if iou > overlap_threshold:
                overlap = True
                break

        if not overlap:
            selected.append((score, x, y))

    return selected


# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------

def load_caption(image_path: Path) -> tuple[str | None, str | None]:
    """Return (txt_content, caption_content) for an image, or None if absent."""
    stem = image_path.with_suffix("")
    txt_path = stem.with_suffix(".txt")
    cap_path = stem.with_suffix(".caption")
    txt = txt_path.read_text(encoding="utf-8").strip() if txt_path.is_file() else None
    cap = cap_path.read_text(encoding="utf-8").strip() if cap_path.is_file() else None
    return txt, cap


def write_caption(dest_stem: Path, txt: str | None, cap: str | None,
                  prefix: str) -> None:
    """Write prefixed captions next to the crop image."""
    if txt is not None:
        content = (prefix + txt) if prefix else txt
        dest_stem.with_suffix(".txt").write_text(content, encoding="utf-8")
    if cap is not None:
        content = (prefix + cap) if prefix else cap
        dest_stem.with_suffix(".caption").write_text(content, encoding="utf-8")
    # If neither exists write a bare trigger-word file so the image isn't
    # captionless (caller can disable this with --no_fallback_caption).
    if txt is None and cap is None and prefix:
        dest_stem.with_suffix(".txt").write_text(prefix.rstrip(", "), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(image_path: Path,
                  output_dir: Path,
                  *,
                  crop_size: int,
                  min_image_size: int,
                  num_crops: int,
                  stride_fraction: float,
                  overlap_threshold: float,
                  score_percentile: float,
                  analysis_radius: int,
                  caption_prefix: str,
                  no_fallback_caption: bool,
                  save_score_map: bool,
                  dry_run: bool) -> int:
    """
    Process a single image.  Returns the number of crops saved.
    """
    try:
        img_pil = Image.open(image_path)
    except Exception as e:
        print(f"  [skip] Cannot open {image_path}: {e}", file=sys.stderr)
        return 0

    W, H = img_pil.size
    if min(W, H) < min_image_size:
        return 0

    # Determine actual crop size (can be smaller than requested)
    cs = min(crop_size, W, H)
    stride = max(1, int(cs * stride_fraction))

    # Texture analysis on grayscale
    gray = np.array(img_pil.convert("L"))
    score_map = compute_texture_score_map(gray, analysis_radius=analysis_radius)

    if save_score_map and not dry_run:
        sm_vis = (score_map * 255).clip(0, 255).astype(np.uint8)
        sm_path = output_dir / (image_path.stem + "_scoremap.png")
        sm_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sm_vis).save(sm_path)

    candidates = score_all_crops(score_map, cs, stride)
    if not candidates:
        return 0

    all_scores = [s for s, _, _ in candidates]
    min_score = float(np.percentile(all_scores, score_percentile))

    selected = greedy_nms(candidates, cs, num_crops, min_score, overlap_threshold)
    if not selected:
        return 0

    txt, cap = load_caption(image_path)

    img_rgb = img_pil.convert("RGB")
    img_np = np.array(img_rgb)

    saved = 0
    for i, (score, x, y) in enumerate(selected):
        dest_name = f"{image_path.stem}_crop{i}"
        dest_dir = output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        crop = img_np[y:y + cs, x:x + cs]
        dest_img = dest_dir / (dest_name + image_path.suffix)

        if not dry_run:
            Image.fromarray(crop).save(dest_img)
            if not no_fallback_caption:
                write_caption(dest_dir / dest_name, txt, cap, caption_prefix)
            elif txt is not None or cap is not None:
                write_caption(dest_dir / dest_name, txt, cap, caption_prefix)

        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def collect_images(input_dir: Path) -> list[Path]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return sorted(paths)


def mirror_subpath(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Reproduce the subdirectory structure of input_dir inside output_dir."""
    rel = image_path.parent.relative_to(input_dir)
    return output_dir / rel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract texture-rich crops from training images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", required=True,
                   help="Root directory containing training images (searched recursively).")
    p.add_argument("--output_dir", default=None,
                   help="Destination directory (default: <input_dir>_texture_crops).")
    p.add_argument("--crop_size", type=int, default=1024,
                   help="Target crop side length in pixels. Reduced automatically if the "
                        "image is smaller.")
    p.add_argument("--num_crops", type=int, default=4,
                   help="Maximum number of crops to extract per image.")
    p.add_argument("--min_image_size", type=int, default=256,
                   help="Skip images whose shortest side is below this value.")
    p.add_argument("--stride_fraction", type=float, default=0.25,
                   help="Sliding-window stride as a fraction of crop_size. "
                        "0.25 = 75%% overlap (thorough); 0.5 = 50%% (faster).")
    p.add_argument("--overlap_threshold", type=float, default=0.25,
                   help="Maximum IoU between two selected crops (NMS threshold). "
                        "Lower = more spread out crops.")
    p.add_argument("--score_percentile", type=float, default=50.0,
                   help="Only consider crops whose score exceeds this percentile of "
                        "all candidate scores. Raise to select only the very richest regions.")
    p.add_argument("--analysis_radius", type=int, default=16,
                   help="Neighbourhood radius (pixels) used when computing local texture "
                        "scores. Smaller = detects finer detail; larger = broader regions.")
    p.add_argument("--caption_prefix", type=str, default="",
                   help="Text prepended to every crop caption, e.g. 'texture detail, '. "
                        "Useful to tag crops as style-only without changing the trigger word.")
    p.add_argument("--no_fallback_caption", action="store_true",
                   help="Do not write a caption file when the source image has no caption. "
                        "By default a bare caption_prefix file is written.")
    p.add_argument("--save_score_map", action="store_true",
                   help="Save a greyscale visualisation of the texture score map alongside "
                        "each image (useful for debugging crop selection).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be done without writing any files.")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        sys.exit(f"Error: input_dir does not exist: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else \
        input_dir.parent / (input_dir.name + "_texture_crops")

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Crop   : {args.crop_size}px  x{args.num_crops} per image  stride={args.stride_fraction}")
    if args.dry_run:
        print("[DRY RUN — no files will be written]")
    print()

    images = collect_images(input_dir)
    if not images:
        sys.exit("No images found.")
    print(f"Found {len(images)} image(s).")

    total_crops = 0
    for img_path in tqdm(images, unit="img"):
        dest_dir = mirror_subpath(img_path, input_dir, output_dir)
        n = process_image(
            img_path,
            dest_dir,
            crop_size=args.crop_size,
            min_image_size=args.min_image_size,
            num_crops=args.num_crops,
            stride_fraction=args.stride_fraction,
            overlap_threshold=args.overlap_threshold,
            score_percentile=args.score_percentile,
            analysis_radius=args.analysis_radius,
            caption_prefix=args.caption_prefix,
            no_fallback_caption=args.no_fallback_caption,
            save_score_map=args.save_score_map,
            dry_run=args.dry_run,
        )
        total_crops += n

    print(f"\nDone. {total_crops} crop(s) {'would be ' if args.dry_run else ''}saved to {output_dir}")


if __name__ == "__main__":
    main()
