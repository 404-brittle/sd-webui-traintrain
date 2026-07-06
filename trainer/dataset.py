from PIL import Image
import glob
import os
import math
from math import gcd
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


def _current_texture_tile_px(t, epoch: int) -> int:
    """Return the active tile side in *pixels* for the given epoch.

    Regime: epoch-based, inclusive of max. Every `texture_tile_step_epochs` epochs,
    the tile size advances by `texture_tile_snap` pixels, clamped to
    [texture_min_tile, texture_max_tile]. The final N epochs (one full step window)
    are spent at max_tile so the schedule is inclusive.
    """
    min_tile = max(8, int(getattr(t, 'texture_min_tile', 256)))
    max_tile = max(min_tile, int(getattr(t, 'texture_max_tile', 1024)))
    snap = max(8, int(getattr(t, 'texture_tile_snap', 128)))
    step_epochs = max(1, int(getattr(t, 'texture_tile_step_epochs', 5)))
    stages = (max_tile - min_tile) // snap
    # stage_index = epoch // step_epochs; clamped so it lands at max for the final stage.
    stage_index = min(stages, max(0, epoch // step_epochs))
    tile = min_tile + snap * stage_index
    return max(min_tile, min(max_tile, tile))


def _current_texture_shift(t, epoch: int) -> float:
    """Return the active flow-shift value for the given epoch.

    The shift advances by one tile-stage per `texture_shift_step_epochs` epochs,
    interpolated linearly between texture_min_shift and texture_max_shift.
    Same inclusivity as the tile schedule.
    """
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
    ratio = stage_index / stages
    return min_shift + (max_shift - min_shift) * ratio


def _parse_aspect_ratios(text: str):
    """Parse ``"2:1,3:1"`` into a list of ``(w_ratio, h_ratio)`` tuples.

    Each entry is ``width:height``.  Both the given ratio and its inverse
    are added (e.g. ``2:1`` → ``(2,1)`` for horizontal + ``(1,2)`` for vertical).
    Square ``1:1`` and invalid entries are skipped.
    Returns an empty list when *text* is empty (backward compatible).
    """
    ratios: list = []
    seen: set = set()
    if not text:
        return ratios
    for part in text.replace(",", " ").split():
        part = part.strip()
        if ":" not in part:
            continue
        try:
            w_str, h_str = part.split(":", 1)
            w, h = int(w_str.strip()), int(h_str.strip())
            if w <= 0 or h <= 0 or w == h:
                continue
            g = gcd(w, h)
            w //= g
            h //= g
            # Add the parsed ratio
            if (w, h) not in seen:
                seen.add((w, h))
                ratios.append((w, h))
            # Add its inverse (vertical ↔ horizontal)
            if (h, w) not in seen:
                seen.add((h, w))
                ratios.append((h, w))
        except (ValueError, ZeroDivisionError):
            continue
    return ratios


def _current_texture_crop_size(t, epoch_now: int, tile_px: int,
                                img_w: int, img_h: int):
    """Return ``(crop_w, crop_h)`` for the given epoch under the aspect-ratio schedule.

    Early epochs always use square crops (``tile_px × tile_px``).  As training
    progresses, non-square aspect ratios from ``t.crop_aspect`` are mixed in with
    linearly increasing probability, reaching 0.5 at the final tile-size stage.

    The crop is computed as the **largest rectangle** at the chosen aspect ratio
    that fits entirely within the image bounds (``scale = min(img_w / w_ratio,
    img_h / h_ratio)``), so it never exceeds the canvas.  Wide images prioritise
    horizontal ratios; tall images prioritise vertical ratios.

    Both dimensions are aligned to 16 px for VAE × DiT patch compatibility.
    When no aspect ratios are configured, always returns the square size
    (backward compatible).
    """
    aspect_text = getattr(t, 'texture_crop_aspect', '') or ''
    all_ratios = _parse_aspect_ratios(aspect_text)
    if not all_ratios:
        return tile_px, tile_px

    # Determine training progress as a fraction 0..1 based on the tile schedule.
    min_tile = max(8, int(getattr(t, 'texture_min_tile', 256)))
    max_tile = max(min_tile, int(getattr(t, 'texture_max_tile', 1024)))
    snap = max(8, int(getattr(t, 'texture_tile_snap', 128)))
    step_epochs = max(1, int(getattr(t, 'texture_tile_step_epochs', 5)))
    stages = (max_tile - min_tile) // snap
    if stages > 0:
        stage_index = min(stages, max(0, epoch_now // step_epochs))
        progress = stage_index / stages
    else:
        progress = 1.0

    # Probability of picking a non-square crop: ramps from 0 → 0.5.
    mix_p = 0.5 * progress

    if random.random() >= mix_p:
        return tile_px, tile_px  # square

    # Prioritise aspect ratios that match the image orientation:
    #   wide image → horizontal ratios (w > h)
    #   tall image → vertical ratios   (h > w)
    is_wide = img_w >= img_h
    candidates = [(w, h) for w, h in all_ratios if (w > h) == is_wide]
    if not candidates:
        candidates = all_ratios

    w_ratio, h_ratio = random.choice(candidates)

    # Largest rectangle at (w_ratio : h_ratio) that fits inside (img_w, img_h).
    scale = min(img_w / w_ratio, img_h / h_ratio)
    crop_w = int(w_ratio * scale)
    crop_h = int(h_ratio * scale)

    # Align to 16px (VAE downsample 8× × DiT spatial_patch_size 2).
    crop_w = max(16, (crop_w // 16) * 16)
    crop_h = max(16, (crop_h // 16) * 16)

    # Re-check fit after alignment (rounding may push us 1 px over).
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    # If the result is effectively square, fall back to square.
    if abs(crop_w - crop_h) < 16:
        return tile_px, tile_px

    return crop_w, crop_h


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

            # Texture-mode JIT source: pull a square tile from the source image
            # at the schedule-driven pixel size, encode it, and use it directly
            # as the full latent (no canvas, no background noise, no feather).
            if isinstance(item, list) and item[0] == "texture_source":
                _, image, mask, emb1, emb2 = item

                # Determine the active tile size in pixels for this epoch.
                # `t.dataloader.epoch` is set by ContinualRandomDataLoader and is
                # incremented every full pass through the dataset.
                epoch_now = getattr(getattr(self.t, 'dataloader', None), 'epoch', 0)
                tile_px_target = _current_texture_tile_px(self.t, epoch_now)
                # Image can be smaller than the target; clamp to image side.
                tile_px_target = min(tile_px_target, image.width, image.height)
                # Align tile size to VAE × DiT patch alignment (16px).
                # The VAE downsamples by 8× (3 stride-2 convs with padding)
                # and the DiT uses spatial_patch_size=2.  If the crop
                # dimension is not a multiple of 16 the resulting latent H/W
                # may be odd, which triggers a PatchEmbed assertion failure
                # (e.g. H,W (87, 87) should be divisible by patch_size 2).
                tile_px_target = (tile_px_target // 16) * 16
                if tile_px_target < 16:
                    tile_px_target = 16

                # Determine aspect-ratio-adjusted crop size for this epoch.
                # Non-square ratios (e.g. 2:1, 3:1) are mixed in with increasing
                # probability as training progresses to reduce square-crop bias.
                crop_w, crop_h = _current_texture_crop_size(
                    self.t, epoch_now, tile_px_target, image.width, image.height,
                )

                best_crop = None
                best_energy = -1
                best_mask_score = -1.0

                mask_np_full = None
                if mask is not None and self.texture_avoid_masked:
                    mask_np_full = np.array(mask.convert("L"), dtype=np.float32) / 255.0

                has_mask   = mask_np_full is not None
                use_energy = self.texture_energy_threshold > 0
                max_attempts = 10 if (has_mask or use_energy) else 1

                best_ms         = -1.0
                best_energy_val = -1.0
                best_region     = (0, 0)

                for attempt in range(max_attempts):
                    src_y = random.randint(0, image.height - crop_h)
                    src_x = random.randint(0, image.width - crop_w)

                    candidate_crop = image.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))

                    mask_score = 1.0
                    if has_mask:
                        m_region = mask_np_full[src_y:src_y + crop_h, src_x:src_x + crop_w]
                        mask_score = float(m_region.mean())

                    energy = 0.0
                    if use_energy and min(crop_w, crop_h) >= 3:
                        gray = np.array(candidate_crop.convert("L"), dtype=np.float32) / 255.0
                        laplace = (
                            gray[1:-1, 1:-1] * 4 -
                            gray[:-2, 1:-1] - gray[2:, 1:-1] -
                            gray[1:-1, :-2] - gray[1:-1, 2:]
                        )
                        if has_mask:
                            m_crop = mask.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
                            m_crop = m_crop.convert("L").resize(
                                (gray.shape[1] - 2, gray.shape[0] - 2), Image.BILINEAR)
                            laplace = laplace * (np.array(m_crop).astype(np.float32) / 255.0)
                        energy = float(np.var(laplace))

                    if mask_score > best_ms or (mask_score == best_ms and energy > best_energy_val):
                        best_ms         = mask_score
                        best_energy_val = energy
                        best_crop = candidate_crop
                        best_region = (src_x, src_y)

                    if mask_score >= 1.0 and (not use_energy or energy >= self.texture_energy_threshold):
                        break

                # Resize the (possibly non-square) crop to the square tile size for
                # consistent latent dimensions across the batch.  Squashing/stretching
                # is acceptable because the model learns invariance to crop framing,
                # which is the goal of mixing aspect ratios.
                crop_pil = best_crop.resize((tile_px_target, tile_px_target), Image.LANCZOS)

                # Encode tile directly. No canvas, no background fill, no feather.
                latent = self.t.image2latent(self.t, crop_pil)  # [1, C, th, tw]

                # Loss mask: cosine-feathered edge taper across `feather_px` latent
                # pixels. Interior is 1.0; loss is computed over the full tile but
                # the edges are slightly downweighted to keep the train target
                # consistent with the padded "all-ones" boundary of the latent.
                th_lat = tile_px_target // 8
                feather = max(0, min(self.texture_feather_latent_px, th_lat // 2))
                loss_mask = torch.ones(th_lat, th_lat)
                if feather > 0:
                    for d in range(feather):
                        v = 0.5 * (1.0 - math.cos(math.pi * d / feather))
                        loss_mask[d, :]      *= v
                        loss_mask[th_lat - 1 - d, :] *= v
                        loss_mask[:, d]      *= v
                        loss_mask[:, th_lat - 1 - d] *= v

                # Blend in alpha mask if available, aligned with the actual region.
                if mask is not None and has_mask:
                    sx, sy = best_region
                    mask_crop = mask.crop((sx, sy, sx + crop_w, sy + crop_h))
                    mask_crop = mask_crop.convert("L").resize((tile_px_target, tile_px_target), Image.BILINEAR)
                    mask_np = np.array(mask_crop).astype(np.float32) / 255.0
                    alpha_lat = torch.from_numpy(
                        F.interpolate(
                            torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0),
                            size=(th_lat, th_lat), mode='bilinear', align_corners=False
                        )[0, 0].numpy()
                    )
                    loss_mask = loss_mask * alpha_lat

                cond1, cond2 = emb1, emb2
                batch["batch_type"] = "texture"
                batch["mask"] = loss_mask.cpu()
            else:
                # Pre-encoded latent path (non-texture): straight forward.
                latent, mask, cond1, cond2 = item
                batch["batch_type"] = "fullres"

            batch["latent"] = latent.squeeze().cpu()
            if cond1 is not None: batch["cond1"] = cond1 if isinstance(cond1, (str, tuple, list)) else cond1.squeeze().cpu()
            if cond2 is not None: batch["cond2"] = cond2 if isinstance(cond2, (str, tuple, list)) else cond2.squeeze().cpu()
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

        # --- Texture mode: store raw PIL sources for JIT tile extraction. ---
        if getattr(t, 'texture_mode', False):
            bucket_key = (t.image_size[0], t.image_size[0])

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
            t.image_buckets_raw[bucket_key].append([
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
                    t.image_buckets[key].append([
                        "texture_source", image, mask,
                        emb1, emb2
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