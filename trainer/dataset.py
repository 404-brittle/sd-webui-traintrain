from PIL import Image
import glob
import os
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import random
import torch.nn.functional as F

test = False

def make_dataloaders(t):
    find_filesets(t)                    #画像、テキスト、キャプションのパスを取得
    make_buckets(t)                     #画像サイズのリストを作成
    load_resize_image_and_text(t)       #画像を読み込み、画像サイズごとに振り分け、リサイズ、テキストの読み込み
                                        #t.image_bucketsは画像サイズをkeyとしたimage,txt, captionのリスト
    encode_image_text(t)                #画像とテキストをlatentとembeddingに変換

    dataloaders = []                    #データセットのセットを作成
    for key in t.image_buckets:
        if test: save_images(t, key, t.image_buckets_raw[key])
        dataset = LatentsConds(t, t.image_buckets[key])
        if dataset.__len__() > 0:
            dataloaders.append(DataLoader(dataset, batch_size=t.train_batch_size, shuffle=True))
        
    return dataloaders

class ContinualRandomDataLoader:
    def __init__(self, dataloaders):
        self.original_dataloaders = dataloaders
        self.epoch = 0
        self.data = len(self.original_dataloaders) > 0
        self._reset_iterators()

    def _reset_iterators(self):
        # すべての DataLoader から新しいイテレータを生成
        self.dataloaders = list(self.original_dataloaders)
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iterators:
            # すべての DataLoader が終了したらリセット
            self._reset_iterators()

        while self.iterators:
            # ランダムに DataLoader を選択
            idx = random.randrange(len(self.iterators))
            try:
                return next(self.iterators[idx])
            except StopIteration:
                # 終了した DataLoader をリストから削除
                self.iterators.pop(idx)
                self.dataloaders.pop(idx)

        # すべての DataLoader が終了した場合
        self.epoch += 1
        raise StopIteration
                                               
def _squeeze_cond(cond):
    """Remove the leading batch dim (size 1) from conditioning tensors so
    DataLoader can re-batch them correctly.  Tuples of tensors (Anima cond
    format) have each element squeezed; plain tensors and strings pass through.
    """
    if isinstance(cond, str):
        return cond
    if isinstance(cond, (tuple, list)):
        squeezed = tuple(
            c.squeeze(0).cpu() if isinstance(c, torch.Tensor) and c.dim() > 1 and c.shape[0] == 1 else c
            for c in cond
        )
        return squeezed
    if isinstance(cond, torch.Tensor):
        return cond.squeeze().cpu()
    return cond


def _freq_matched_background(tile: torch.Tensor, canvas_h: int, canvas_w: int) -> torch.Tensor:
    """Generate a canvas-sized background whose power spectrum matches `tile` but with
    completely randomised spatial phase.

    The tile has real spatial structure (correlated adjacent latent pixels).  Plain
    i.i.d. N(μ,σ²) noise has no such correlations.  At intermediate timesteps (~300–700)
    the model can detect the "correlated region vs. uncorrelated region" boundary in the
    forward-pass input even though the loss mask zeroes it out, causing boundary/tiling
    artefacts in the trained LoRA.

    By matching the power spectrum we make the background statistically indistinguishable
    from texture content: same spatial frequency distribution, different specific content
    every call.  The model's attention cannot consistently find a boundary.

    Concretely: take the per-channel 2-D FFT of the tile, resample the magnitude spectrum
    to canvas size, multiply by a fresh unit-complex random phase, then IFFT.
    """
    C = tile.shape[0]

    # Per-channel FFT of the tile (real-to-complex)
    tile_fft = torch.fft.rfft2(tile.float())           # [C, th, tw//2+1]  complex
    tile_mag = tile_fft.abs()                          # [C, th, tw//2+1]  real

    # Resample tile magnitude spectrum to canvas FFT dimensions
    tile_mag_resized = F.interpolate(
        tile_mag.unsqueeze(0),                         # [1, C, th, tw//2+1]
        size=(canvas_h, canvas_w // 2 + 1),
        mode='bilinear', align_corners=False,
    ).squeeze(0)                                       # [C, canvas_h, canvas_w//2+1]

    # Random unit-complex phase (fresh every call) — on same device as tile
    rand_angle = torch.rand(C, canvas_h, canvas_w // 2 + 1, device=tile_mag_resized.device) * (2 * math.pi)
    rand_phase = torch.polar(torch.ones_like(rand_angle), rand_angle)  # unit complex

    # Apply magnitude from tile, random phase
    result_fft = tile_mag_resized * rand_phase
    result = torch.fft.irfft2(result_fft, s=(canvas_h, canvas_w))      # [C, H, W]

    return result.to(tile.dtype)


def _place_texture_crop(crop_latent: torch.Tensor, canvas_hw: tuple, feather_px: int,
                        alpha_crop: torch.Tensor | None = None,
                        tile_px: int | None = None):
    """Place a square tile from the (pre-scaled) image latent at a random position on a
    canvas whose background is filled with frequency-matched noise, and return a feathered
    mask that is non-zero only over the tile region.

    Background fill uses _freq_matched_background: noise with the tile's power spectrum
    but randomised spatial phase.  This makes the background statistically
    indistinguishable from real texture content — same spatial correlation structure,
    different specific values every call.  Plain N(μ,σ²) noise lacks spatial correlations
    and creates a detectable "correlated vs. uncorrelated" boundary in the forward pass at
    intermediate timesteps (~300–700), even though the loss mask zeroes it out, leading to
    boundary/tiling artefacts in the trained LoRA.  Frequency-matched noise eliminates
    this boundary at all timestep levels, allowing the full 0–1000 range without
    positional or boundary leakage.

    The mask remains zero outside the tile: the background has no learnable target, so
    including it in the loss would only add gradient variance.

    Both the source position within the image latent and (when tile_px is None) tile size
    are re-sampled every call, giving positional/scale variance across DataLoader draws.

    Args:
        crop_latent: encoded latent of the (already tile-scaled) image, any size.
        canvas_hw:   (H, W) in pixels of the training canvas.
        feather_px:  cosine feather width in latent pixels.
        alpha_crop:  optional [img_lat_h, img_lat_w] float tensor (0-1) aligned with
                     the full image latent. Multiplied into the feathered patch.
        tile_px:     if given, the fixed square tile side in *pixels* (divided by 8
                     for latent space). The tile is clamped to the image latent size
                     when the image is smaller. When None, tile size is random.

    Returns:
        canvas  — [1, C, canvas_lat_h, canvas_lat_w]  float tensor
        mask    — [1, canvas_lat_h, canvas_lat_w]      float tensor in [0, 1]
    """
    canvas_h, canvas_w = canvas_hw
    clat_h, clat_w = canvas_h // 8, canvas_w // 8

    crop = crop_latent.squeeze(0)          # [C, img_h, img_w]
    _, ih, iw = crop.shape

    # --- Determine square tile size in latent space ---
    if tile_px is not None:
        # Fixed tile: convert pixels → latent units, clamp to what fits in both
        # the (scaled) image latent and the canvas latent.
        tile_lat = max(1, tile_px // 8)
        th = tw = min(tile_lat, ih, iw, clat_h, clat_w)
    else:
        # Random tile: uniform in [max/4, max] where max fits both image and canvas.
        max_tile = min(ih, iw, clat_h, clat_w)
        min_tile = max(1, max_tile // 4)
        th = tw = random.randint(min_tile, max_tile)

    # Random source position within the image latent
    src_y = random.randint(0, ih - th)
    src_x = random.randint(0, iw - tw)
    tile = crop[:, src_y:src_y + th, src_x:src_x + tw]   # [C, th, tw]

    # --- Random destination position within the canvas ---
    max_oy = clat_h - th
    max_ox = clat_w - tw
    oy = random.randint(0, max_oy)
    ox = random.randint(0, max_ox)

    # Fill background with frequency-matched noise: same power spectrum as the tile
    # but randomised spatial phase, so the background looks like texture-like content
    # with no consistent boundary the model can detect or memorise.
    canvas = _freq_matched_background(tile, clat_h, clat_w)
    canvas[:, oy:oy + th, ox:ox + tw] = tile

    # Feathered mask: cosine taper over `feather` latent pixels from every edge
    feather = min(feather_px, th // 2, tw // 2)
    patch = torch.ones(th, tw)
    for d in range(feather):
        v = 0.5 * (1.0 - math.cos(math.pi * d / feather))
        patch[d, :]      *= v   # top edge
        patch[th-1-d, :] *= v   # bottom edge
        patch[:, d]      *= v   # left edge
        patch[:, tw-1-d] *= v   # right edge

    # Blend in alpha mask (from image alpha channel or external mask file).
    # alpha_crop is aligned with the full image latent, so crop the same region.
    if alpha_crop is not None:
        ac = alpha_crop
        if ac.shape != (ih, iw):
            ac = F.interpolate(ac.unsqueeze(0).unsqueeze(0).float(),
                               size=(ih, iw), mode='bilinear', align_corners=False)[0, 0]
        patch = patch * ac[src_y:src_y + th, src_x:src_x + tw]

    mask = torch.zeros(1, clat_h, clat_w)
    mask[0, oy:oy + th, ox:ox + tw] = patch

    return canvas.unsqueeze(0), mask   # [1,C,H,W], [1,H,W]


class LatentsConds(Dataset):
    def __init__(self, t, latents_conds):
        self.t = t
        self.latents_conds = latents_conds
        self.batch_size = t.train_batch_size
        self.revert = t.diff_revert_original_target
        self.texture_feather_latent_px = getattr(t, 'texture_feather_latent_px', 2)
        self.texture_energy_threshold = getattr(t, 'texture_energy_threshold', 0)
        self.texture_avoid_masked = getattr(t, 'texture_avoid_masked', True)
        # image_num_multiply hardcoded to 1
        if t.train_batch_size > len(self.latents_conds):
            self.latents_conds = self.latents_conds * t.train_batch_size

    def __len__(self):
        return len(self.latents_conds)

    def __getitem__(self, i):
        batch = {}
        if isinstance(self.latents_conds[i], tuple):
            origs, targs = self.latents_conds[i]
            if self.revert:
                targs, origs = origs, targs
            orig_latent, orig_mask, orig_cond1, orig_cond2 = origs  
            targ_latent, targ_mask, targ_cond1, targ_cond2 = targs

            batch["orig_latent"] = orig_latent.squeeze()
            batch["targ_latent"] = targ_latent.squeeze()
            if orig_cond1 is not None: batch["orig_cond1"] = _squeeze_cond(orig_cond1)
            if orig_cond2 is not None: batch["orig_cond2"] = _squeeze_cond(orig_cond2)
            if targ_cond1 is not None: batch["targ_cond1"] = _squeeze_cond(targ_cond1)
            if targ_cond2 is not None: batch["targ_cond2"] = _squeeze_cond(targ_cond2)
            if isinstance(orig_mask, torch.Tensor): batch["mask"] = orig_mask.squeeze().cpu()

        else:
            item = self.latents_conds[i]
            
            # Detect JIT texture source
            if isinstance(item, list) and item[0] == "texture_source":
                _, image, mask, emb1, emb2, canvas_hw, tile_res, tile_scale = item

                _TEXTURE_CANVAS_PRESETS = [
                    (640, 1536), (1536, 640),
                    (832, 1216), (1216, 832),
                    (1024, 1024), (1024, 1024), #twice, for 1/3 chance.
                ]

                # Hybrid fullres mode: resize entire image to a canvas matching its
                # aspect ratio and encode — no crop, no mask, no canvas noise.
                if getattr(self.t, 'hybrid_processing_mode', None) == "fullres":
                    img_ar = image.width / image.height
                    canvas_hw = min(_TEXTURE_CANVAS_PRESETS,
                                    key=lambda hw: abs(hw[1] / hw[0] - img_ar))
                    canvas_h, canvas_w = canvas_hw
                    # Scale to cover the canvas (preserve AR), then center-crop the
                    # overhang.  This cuts a tiny sliver of data rather than squashing.
                    scale = max(canvas_w / image.width, canvas_h / image.height)
                    scaled_w = round(image.width * scale)
                    scaled_h = round(image.height * scale)
                    img_resized = image.resize((scaled_w, scaled_h), Image.LANCZOS)
                    left = (scaled_w - canvas_w) // 2
                    top  = (scaled_h - canvas_h) // 2
                    img_resized = img_resized.crop((left, top, left + canvas_w, top + canvas_h))
                    latent = self.t.image2latent(self.t, img_resized)
                    mask = None
                    cond1, cond2 = emb1, emb2
                    batch["batch_type"] = "fullres"
                    batch["latent"] = latent.squeeze().cpu()
                    if cond1 is not None: batch["cond1"] = cond1 if isinstance(cond1, (str, tuple, list)) else cond1.squeeze().cpu()
                    if cond2 is not None: batch["cond2"] = cond2 if isinstance(cond2, (str, tuple, list)) else cond2.squeeze().cpu()
                    return batch

                # Randomise canvas resolution each step for aspect-ratio invariance
                canvas_hw = random.choice(_TEXTURE_CANVAS_PRESETS)

                # JIT: Randomly crop, scale, and encode
                canvas_h, canvas_w = canvas_hw
                clat_h, clat_w = canvas_h // 8, canvas_w // 8
                
                best_crop = None
                best_energy = -1
                best_mask_score = -1.0

                # Pre-compute full-resolution mask as numpy for fast per-crop mean.
                # Only built when texture_avoid_masked is set; None disables all
                # mask scoring so the loop runs just once and takes the first crop.
                mask_np_full = None
                if mask is not None and self.texture_avoid_masked:
                    mask_np_full = np.array(mask.convert("L"), dtype=np.float32) / 255.0

                has_mask   = mask_np_full is not None
                use_energy = self.texture_energy_threshold > 0
                max_attempts = 10 if (has_mask or use_energy) else 1

                best_ms         = -1.0
                best_energy_val = -1.0

                for attempt in range(max_attempts):
                    # Resolve tile size in pixels
                    if tile_res > 0:
                        tile_px = tile_res
                    else:
                        max_tile_lat = min(image.width // 8, image.height // 8, clat_h, clat_w)
                        if tile_scale > 1.0:
                            max_tile_lat = min(max_tile_lat, round(image.width * tile_scale) // 8, round(image.height * tile_scale) // 8)
                        min_tile_lat = max(1, max_tile_lat // 4)
                        tile_lat = random.randint(min_tile_lat, max_tile_lat)
                        tile_px = tile_lat * 8

                    src_px = max(8, round(tile_px / tile_scale))
                    src_px = min(src_px, image.width, image.height)
                    tile_px = round(src_px * tile_scale)

                    src_y = random.randint(0, image.height - src_px)
                    src_x = random.randint(0, image.width - src_px)

                    candidate_crop = image.crop((src_x, src_y, src_x + src_px, src_y + src_px))

                    # Mask score: 1.0 = no masked pixels, <1.0 = some masked pixels
                    mask_score = 1.0
                    if has_mask:
                        m_region = mask_np_full[src_y:src_y + src_px, src_x:src_x + src_px]
                        mask_score = float(m_region.mean())

                    # Energy: Laplacian variance; mask zeroes out masked contributions
                    energy = 0.0
                    if use_energy:
                        gray = np.array(candidate_crop.convert("L"), dtype=np.float32) / 255.0
                        laplace = (
                            gray[1:-1, 1:-1] * 4 -
                            gray[:-2, 1:-1] - gray[2:, 1:-1] -
                            gray[1:-1, :-2] - gray[1:-1, 2:]
                        )
                        if has_mask:
                            m_crop = mask.crop((src_x, src_y, src_x + src_px, src_y + src_px))
                            m_crop = m_crop.convert("L").resize((src_px - 2, src_px - 2), Image.BILINEAR)
                            laplace = laplace * (np.array(m_crop).astype(np.float32) / 255.0)
                        energy = float(np.var(laplace))

                    # Update best: mask coverage PRIMARY, energy secondary tiebreak
                    if mask_score > best_ms or (mask_score == best_ms and energy > best_energy_val):
                        best_ms         = mask_score
                        best_energy_val = energy
                        best_crop = (candidate_crop, src_x, src_y, src_px, tile_px)

                    # Early accept: zero masked pixels AND energy threshold satisfied
                    if mask_score >= 1.0 and (not use_energy or energy >= self.texture_energy_threshold):
                        break

                
                crop_pil, src_x, src_y, src_px, tile_px = best_crop
                crop_pil = crop_pil.resize((tile_px, tile_px), Image.LANCZOS)
                
                # Encode tile
                latent = self.t.image2latent(self.t, crop_pil) # [1, C, th, tw]
                
                # Handle alpha mask if present
                alpha_crop_mask = None
                if mask is not None:
                    mask_crop = mask.crop((src_x, src_y, src_x + src_px, src_y + src_px)).resize((tile_px, tile_px), Image.BILINEAR)
                    mask_np = np.array(mask_crop).astype(np.float32) / 255.0
                    alpha_crop_mask = torch.from_numpy(mask_np) # [th, tw]
                
                # Place and feather
                latent, mask = _place_texture_crop(
                    latent, canvas_hw, self.texture_feather_latent_px,
                    alpha_crop=alpha_crop_mask,
                    tile_px=tile_px
                )
                
                cond1, cond2 = emb1, emb2
                batch["batch_type"] = "texture"
            else:
                if len(item) == 6:
                    latent, alpha_crop_mask, cond1, cond2, canvas_hw, tile_res = item
                elif len(item) == 5:
                    latent, alpha_crop_mask, cond1, cond2, canvas_hw = item
                    tile_res = 0
                else:
                    latent, mask, cond1, cond2 = item
                    canvas_hw = None
                    alpha_crop_mask = None
                    tile_res = 0

                if canvas_hw is not None:
                    # Texture mode (legacy/pre-encoded): sample tile then place
                    latent, mask = _place_texture_crop(
                        latent, canvas_hw, self.texture_feather_latent_px,
                        alpha_crop=alpha_crop_mask,
                        tile_px=tile_res if tile_res > 0 else None,
                    )
                    batch["batch_type"] = "texture"
                else:
                    batch["batch_type"] = "fullres"

            batch["latent"] = latent.squeeze().cpu()
            if cond1 is not None: batch["cond1"] = cond1 if isinstance(cond1, (str, tuple, list)) else cond1.squeeze().cpu()
            if cond2 is not None: batch["cond2"] = cond2 if isinstance(cond2, (str, tuple, list)) else cond2.squeeze().cpu()
            if isinstance(mask, torch.Tensor): batch["mask"] = mask.squeeze().cpu()
        return batch

TARGET_IMAGEFILES = ["jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp", "webp", "pcx", "ico"]

def make_buckets(t):
    increment = t.image_buckets_step # default : 256
    # 最大ピクセル数 resolutionは[x ,y]の配列。 y >= x
    max_pixels = t.image_size[0]*t.image_size[1] 

    # 正方形は手動で追加
    max_buckets = set()
    max_buckets.add((t.image_size[0], t.image_size[0]))

    # 最小値から～
    width = t.image_min_length
    # ～最大値まで
    while width <= max(t.image_size):
        # 最大ピクセル数と最大長を越えない最大の高さ
        height = min(max(t.image_size), (max_pixels // width) - (max_pixels // width) % increment)
        ratio = width/height

        # アスペクト比が極端じゃなかったら追加、高さと幅入れ替えたものも追加。
        if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
            max_buckets.add((width, height))
            max_buckets.add((height, width))
        width += increment  # 幅を大きくして次のループへ

    sub_buckets = set()

    # 最小サイズから最大サイズまでの範囲で枠を生成
    for width in range(t.image_min_length, max(t.image_size) + 1, increment):
        for height in range(t.image_min_length, max(t.image_size) + 1, increment):
            if width * height <= max_pixels:
                ratio = width / height
                if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
                    if (width, height) not in max_buckets:
                        sub_buckets.add((width, height))
                    if (height, width) not in max_buckets:
                        sub_buckets.add((height, width))

    # アスペクト比に基づいて枠を並べ替え
    max_buckets = list(max_buckets)
    max_ratios = [w / h for w, h in max_buckets]
    max_buckets = np.array(max_buckets)[np.argsort(max_ratios)]
    max_buckets = [tuple(x) for x in max_buckets]
    max_ratios = np.sort(max_ratios)

    sub_buckets = list(sub_buckets)
    sub_ratios = [w / h for w, h in sub_buckets]
    sub_buckets = np.array(sub_buckets)[np.argsort(sub_ratios)]
    sub_buckets = [tuple(x) for x in sub_buckets]
    sub_ratios = np.sort(sub_ratios)

    t.image_max_buckets_sizes = max_buckets
    t.image_max_ratios = max_ratios
    t.image_sub_buckets_sizes = sub_buckets
    t.image_sub_ratios = sub_ratios
    t.image_buckets_raw = {}
    t.image_buckets = {}
    print("max bucket sizes : ", max_buckets)
    #t.db("max bucket sizes : ", max_ratios)
    print("sub bucket sizes : ", sub_buckets)
    #t.db("sub bucket sizes : ", sub_ratios)
    for bucket in max_buckets + sub_buckets:
        t.image_buckets_raw[bucket] = []
        t.image_buckets[bucket] = []

def find_filesets(t):
    """
    Create two lists: 
    1. Absolute paths of image files in the specified folder and subfolders.
    2. Absolute paths of corresponding text files, or 'None' if no corresponding text file exists.

    :param folder_path: Path to the folder to search in.
    :param image_extensions: List of image file extensions to look for.
    :return: Tuple of two lists (image_paths, text_paths)
    """
    pathsets = []
    
    # Walk through the folder and subfolders
    pathdict = {}
    for root, dirs, files in os.walk(t.lora_data_directory):
        for file in files:
            if any(file.endswith(ext) for ext in TARGET_IMAGEFILES):
                image_path = os.path.join(root, file)

                filename = os.path.splitext(os.path.basename(image_path))[0]
                filename = filename.split("_id_")[0]
                filename = filename.replace("_", ",")  

                # Check for corresponding text file
                text_file = os.path.splitext(image_path)[0] + '.txt'
                text_file = text_file if os.path.isfile(text_file) else None

                # Check for corresponding caption file
                caption_file = os.path.splitext(image_path)[0] + '.caption'
                caption_file = caption_file if os.path.isfile(caption_file) else None
                pathsets.append([image_path, text_file, caption_file, filename, None, None])
                pathdict[image_path] = [image_path, text_file, caption_file, filename]

    t.db("Images : ", len(pathsets))
    t.db("Texts : " , sum(1 for patch in pathsets if patch[1] is not None))
    t.db("Captions : " , sum(1 for patch in pathsets if patch[2] is not None))

    t.image_pathsets = pathsets

    if t.mode == "Multi-ADDifT":
        pairpathsets = []
        for image_path, _, _, _, _, _ in pathsets:
            base_name, ext = os.path.splitext(image_path)
            diff_target_path = f"{base_name}{t.diff_target_name}{ext}"
            if diff_target_path in pathdict:
                with Image.open(pathdict[image_path][0]) as orig_img:
                    orig_size = orig_img.size
                with Image.open(pathdict[diff_target_path][0]) as targ_img:
                    targ_size = targ_img.size
                    new_size = (min(orig_size[0], targ_size[0]), min(orig_size[1], targ_size[1]))

                pairpathsets.append(pathdict[image_path] + [new_size, pathdict[diff_target_path][0]])
                pairpathsets.append(pathdict[diff_target_path] + [new_size, None])
        t.image_pathsets = pairpathsets


def load_resize_image_and_text(t):
    for img_path, txt_path, cap_path, filename, pair_size, targ_path in t.image_pathsets:
        if os.path.basename(img_path).startswith('.'):
            continue
        image = Image.open(img_path)
        usealpha = image.mode == "RGBA"

        if pair_size is not None and image.size != pair_size:
            image = image.resize(pair_size, Image.LANCZOS)

        # --- Texture mode OR hybrid mode (all images stored as JIT PIL sources) ---
        if getattr(t, 'texture_mode', False) or getattr(t, 'train_hybrid_mode', False):
            canvas_key = (t.image_size[0], t.image_size[0])

            # JIT UPDATE: Store raw PIL objects
            mask_img = None
            if image.mode == "RGBA":
                mask_img = image.split()[3]
            else:
                mask_dir = getattr(t, 'texture_mask_directory', None)
                if mask_dir:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    for _ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
                        _mp = os.path.join(mask_dir, stem + _ext)
                        if os.path.isfile(_mp):
                            mask_img = Image.open(_mp).convert("L").resize(
                                (image.width, image.height), Image.LANCZOS)
                            break
            
            image = image.convert("RGB")
            t.image_buckets_raw[canvas_key].append([
                image, mask_img,
                load_text_files(txt_path), load_text_files(cap_path),
                filename, img_path, targ_path, True,
            ])
            continue

        # buckets logic for non-texture mode...
        ratio = image.width / image.height
        ar_errors = t.image_max_ratios - ratio
        indice = np.argmin(np.abs(ar_errors))
        max_size = t.image_max_buckets_sizes[indice]
        ar_error = ar_errors[indice]

        def resize_and_crop(ar_error, image, bucket_width, bucket_height, disable_upscale):
            if (ar_error > 0 and image.width < bucket_width or 
                ar_error <= 0 and image.height < bucket_height) and disable_upscale:
                return None

            if ar_error <= 0:
                temp_width = int(image.width*bucket_height/image.height)
                image = image.resize((temp_width, bucket_height))
                left = (temp_width - bucket_width) / 2
                right = bucket_width + left
                image = image.crop((left, 0, right, bucket_height))
            else:
                temp_height = int(image.height*bucket_width/image.width)
                image = image.resize((bucket_width, temp_height))
                upper = (temp_height - bucket_height) / 2
                lower = bucket_height + upper
                image = image.crop((0, upper, bucket_width, lower))

            if usealpha:
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                alpha_channel = tensor[3]
                alpha_mask = (alpha_channel > 0.1).float()
                H, W = alpha_mask.shape
                new_H, new_W = H // 8, W // 8
                mask = F.interpolate(alpha_mask.unsqueeze(0).unsqueeze(0), size=(new_H, new_W), mode='nearest')[0, 0]
            else:
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                _, H, W = tensor.shape
                new_H, new_W = H // 8, W // 8
                mask = torch.ones((new_H, new_W))

            image = image.convert("RGB")
            return image, mask

        resized, alpha_mask = resize_and_crop(ar_error, image, *max_size, t.image_disable_upscale)
        if resized is not None:
            t.image_buckets_raw[max_size].append([resized, alpha_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path, targ_path])
            if t.image_mirroring:
                flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_mask = torch.flip(alpha_mask, [1]) if alpha_mask is not None else None
                t.image_buckets_raw[max_size].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path+"m", targ_path+"m" if targ_path is not None else targ_path])

        ar_errors = t.image_sub_ratios - ratio
        try:
            for _ in range(t.sub_image_num):
                idx = np.argmin(np.abs(ar_errors))
                sub = t.image_sub_buckets_sizes[idx]
                err = ar_errors[idx]
                res, msk = resize_and_crop(err, image, *sub, t.image_disable_upscale)
                if res is not None:
                    t.image_buckets_raw[sub].append([res, msk, load_text_files(txt_path), load_text_files(cap_path), filename, img_path, targ_path])
                    if t.image_mirroring:
                        flipped = res.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_mask = torch.flip(msk, [1]) if msk is not None else None
                        t.image_buckets_raw[sub].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path+"m", targ_path+"m" if targ_path is not None else targ_path])
                ar_errors[idx] += 1
        except:
            pass

    for key in t.image_buckets_raw:
        count = len(t.image_buckets_raw[key])
        if count > 0:
            print(f"bucket {key} has {count} images")
        t.total_images += count
    
def load_text_files(file_path):
    if file_path is None:
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def encode_image_text(t):
    with torch.no_grad(), t.a.autocast():
        emp1, emp2 = t.text_model.encode_text(t.lora_trigger_word)
        bar = tqdm(total = t.total_images)
        for key in t.image_buckets_raw:
            pairdict = {}
            for entry in t.image_buckets_raw[key]:
                is_texture = len(entry) == 8 and entry[7]
                image, mask, text, caption, filename, img_path, targ_path = entry[:7]

                if not is_texture:
                    latent = t.image2latent(t, image)
                else:
                    latent = None

                if t.image_use_filename_as_tag:
                    prompt = t.lora_trigger_word + "," + filename
                elif text is not None:
                    prompt = t.lora_trigger_word + ", " + text
                elif caption is not None:
                    prompt = t.lora_trigger_word + ", " + caption
                else:
                    prompt = t.lora_trigger_word
                t.tagcount(prompt)
                if "BASE" not in t.network_blocks:
                    emb1, emb2 = (emp1, emp2) if prompt is None else t.text_model.encode_text(prompt)
                else:
                    emb1 = emb2 = prompt

                if is_texture:
                    canvas_hw = (key[1], key[0])
                    tile_res = getattr(t, 'texture_tile_resolution', 0)
                    tile_scale = getattr(t, 'texture_tile_scale', 1.0)
                    t.image_buckets[key].append([
                        "texture_source", image, mask,
                        emb1, emb2, canvas_hw, tile_res, tile_scale
                    ])
                else:
                    t.image_buckets[key].append([latent, mask, emb1, emb2])
                bar.update(1)
                pairdict[img_path] = [latent, mask, emb1, emb2, targ_path, image]
            
            if t.mode == "Multi-ADDifT":
                t.image_buckets[key] = []
                for img_path_key in pairdict:
                    if pairdict[img_path_key][4] in pairdict:
                        if getattr(t, 'diff_use_diff_mask', False):
                            image_o = pairdict[img_path_key][5]
                            image_t = pairdict[pairdict[img_path_key][4]][5]

                            image_np = np.array(image_o, dtype=np.int16)
                            image_t_np = np.array(image_t, dtype=np.int16)

                            mask = image_np - image_t_np
                            mask = torch.tensor(mask, dtype=torch.float32)
                            mask = mask.abs().sum(dim=-1)
                            mask = torch.where(mask > 10, torch.tensor(1, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))

                            mask = mask.float()
                            dilation = 33
                            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).cuda(), kernel_size=dilation, stride=1, padding=dilation // 2)[0, 0].cpu()
                            save_image1(t, mask * 255, "mask", name=img_path_key)
                            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent.shape[2], latent.shape[3]), mode='nearest')[0, 0]
                            pairdict[img_path_key][1] = mask
                        t.image_buckets[key].append((pairdict[img_path_key][:-2], pairdict[pairdict[img_path_key][4]][:-2]))

def save_images(t,key,images):
    if not images: return
    path = os.path.join(t.lora_data_directory,"x".join(map(str, list(key))))
    os.makedirs(path, exist_ok=True)
    for i, image in enumerate(images):
        ipath = os.path.join(path, f"{i}.jpg")
        image[0].save(ipath)


def save_image1(t, image, dirname="", name=None):
    path = os.path.join(t.lora_data_directory, dirname) if dirname else t.lora_data_directory
    os.makedirs(path, exist_ok=True)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)

        if image.ndim == 2:
            image = Image.fromarray(image.astype(np.uint8), mode='L')
        elif image.ndim == 3:
            if image.shape[0] in [3, 4]:
                image = np.moveaxis(image, 0, -1)
            image = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    try:
        if name is not None:
            stem = os.path.splitext(os.path.basename(name))[0]
            image_path = os.path.join(path, f"{stem}_mask.png")
        else:
            existing = glob.glob(os.path.join(path, "mask_*.png"))
            idx = len(existing)
            image_path = os.path.join(path, f"mask_{idx:04d}.png")
        image.save(image_path)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")