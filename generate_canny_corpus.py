"""
Generate diverse anime images for Canny ControlNet training corpus.
Uses the Forge Neo API to produce varied images with randomised prompts and sizes.
"""

import base64
import random
import argparse
import requests
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

API_URL = "http://127.0.0.1:7861/sdapi/v1/txt2img"

NEGATIVE_PROMPT = (
    "bad hands, missing finger, bad anatomy,"
    "sketch, wip, unfinished."
    "chromatic aberration."
    "signature, watermark, artist name."
    "censored, censor bar."
    "artist name,"
    "text, sound effects, speech bubble,"
    "retro graphics, lowpoly, blocky graphics, pixel art."
    "monochrome, greyscale, muted color,"
    "(blurry, jpeg artifacts:3)"
)

VALID_SIZES = [
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (1536, 640),
    (640, 1536),
]
VALID_SIZES = [
    (512, 512),
    (512, 768),
    (768, 512),
    (640, 640),
]

# --- Prompt building blocks ---

SUBJECTS = [
    "1girl", "1boy", "2girls", "1girl, 1boy",
]

HAIR_COLORS = [
    "black hair", "blonde hair", "white hair", "silver hair", "pink hair",
    "blue hair", "purple hair", "brown hair", "red hair", "green hair",
    "gradient hair", "multicolored hair",
]

HAIR_STYLES = [
    "long hair", "short hair", "twin tails", "ponytail", "braid",
    "side ponytail", "bob cut", "messy hair", "flowing hair",
]

EYE_COLORS = [
    "blue eyes", "red eyes", "green eyes", "golden eyes", "purple eyes",
    "heterochromia", "amber eyes", "gray eyes",
]

EXPRESSIONS = [
    "smile", "serious", "surprised", "shy", "determined",
    "sad", "happy", "expressionless", "blushing",
]

OUTFITS = [
    "school uniform", "sailor uniform", "kimono", "maid outfit",
    "casual clothes", "fantasy armor", "shrine maiden outfit",
    "hoodie", "dress", "jacket", "coat", "sportswear", "military uniform",
    "gothic lolita", "yukata",
]

VIEW_ANGLES = [
    "from above", "from below", "from side", "from behind",
    "dutch angle", "eye level", "low angle", "high angle",
]

COMPOSITIONS = [
    "portrait", "upper body", "full body", "close-up", "bust",
    "cowboy shot", "head shot",
]

POSES = [
    "standing", "sitting", "running", "walking", "looking at viewer",
    "looking away", "arms behind back", "arms crossed", "reaching out",
    "lying down", "crouching", "leaning forward", "jumping",
    "hand on hip", "waving", "reading book", "holding sword",
]

BACKGROUNDS = [
    "simple background", "white background", "outdoor, sky background",
    "city background, buildings", "forest background", "ocean background",
    "cherry blossom background", "classroom background",
    "night sky background, stars", "sunset background",
    "bedroom background", "fantasy landscape", "rainy background",
    "snowy background", "temple background", "rooftop background",
    "gradient background",
]

LIGHTING = [
    "soft lighting", "dramatic lighting", "rim lighting",
    "backlit", "warm lighting", "cool lighting", "sunlight",
    "ambient lighting",
]

QUALITY_TAGS = [
    "masterpiece", "best quality", "highly detailed", "sharp focus",
    "intricate details", "beautiful", "amazing quality",
]


def load_artists(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artists file not found: {path}")
    artists = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    if not artists:
        raise ValueError("Artists file is empty")
    return artists


def build_prompt(artists: list[str]) -> str:
    parts = []

    # Quality
    parts += random.sample(QUALITY_TAGS, k=random.randint(2, 4))

    # Subject
    parts.append(random.choice(SUBJECTS))

    # Physical traits
    parts.append(random.choice(HAIR_COLORS))
    parts.append(random.choice(HAIR_STYLES))
    parts.append(random.choice(EYE_COLORS))
    parts.append(random.choice(EXPRESSIONS))

    # Outfit
    parts.append(random.choice(OUTFITS))

    # Composition & camera
    parts.append(random.choice(COMPOSITIONS))
    if random.random() < 0.5:
        parts.append(random.choice(VIEW_ANGLES))

    # Pose
    parts.append(random.choice(POSES))

    # Background
    parts.append(random.choice(BACKGROUNDS))

    # Lighting (optional)
    if random.random() < 0.4:
        parts.append(random.choice(LIGHTING))

    # Artist
    artist = random.choice(artists)
    parts.append(f"art by {artist}")

    return ", ".join(parts)


def generate_image(prompt: str, width: int, height: int) -> bytes:
    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": -1,
        "sampler_name": "Euler a",
        "scheduler": "Simple",
        "batch_size": 1,
        "steps": 30,
        "cfg_scale": 4,
        "width": width,
        "height": height,
        "do_not_save_samples": True,
    }
    resp = requests.post(API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    img_b64 = data["images"][0]
    return base64.b64decode(img_b64)


def main():
    parser = argparse.ArgumentParser(description="Generate anime corpus for Canny ControlNet training")
    parser.add_argument("--count", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="corpus_output", help="Output directory")
    parser.add_argument("--artists", type=str, default="artists.txt", help="Path to artists list")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    artists = load_artists(args.artists)
    print(f"Loaded {len(artists)} artists from {args.artists}")
    print(f"Generating {args.count} images into {out_dir}/")

    for i in range(args.count):
        width, height = random.choice(VALID_SIZES)
        prompt = build_prompt(artists)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        basename = f"{timestamp}_{i:05d}"
        img_path = out_dir / f"{basename}.png"

        try:
            img_bytes = generate_image(prompt, width, height)

            # Decode image and generate canny edge map
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 100, 200)
            canny_path = out_dir / f"{basename}_canny.png"
            cv2.imwrite(str(canny_path), canny)

            # Save original image
            img_path.write_bytes(img_bytes)

            # Save prompt to matching text file
            txt_path = out_dir / f"{basename}.txt"
            txt_path.write_text(prompt, encoding="utf-8")

            print(f"[{i+1}/{args.count}] {img_path.name} ({width}x{height})")
        except Exception as e:
            print(f"[{i+1}/{args.count}] FAILED: {e}")


if __name__ == "__main__":
    main()
