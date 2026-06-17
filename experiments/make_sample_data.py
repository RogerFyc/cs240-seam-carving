"""Create a small sample image dataset using scikit-image.

Run:
    python experiments/make_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from skimage import data
from src.utils import save_image, ensure_dir


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (255 * img).round().astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def main() -> None:
    output_dir = ensure_dir(ROOT / "data" / "input")

    samples = {
        "astronaut.png": data.astronaut(),
        "coffee.png": data.coffee(),
        "chelsea.png": data.chelsea(),
        "rocket.png": data.rocket(),
        "camera.png": data.camera(),
    }

    for name, img in samples.items():
        path = output_dir / name
        save_image(to_rgb_uint8(img), path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
