"""Utility functions for image I/O and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def image_to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (255.0 * image).round().astype(np.uint8)


def overlay_vertical_seam(image: np.ndarray, seam: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    if len(seam) != h:
        raise ValueError(f"Seam length {len(seam)} does not match image height {h}.")
    for i, j in enumerate(seam):
        if 0 <= j < w:
            out[i, j] = color
    return out


def save_comparison_grid(images: list[np.ndarray], titles: list[str], path: str | Path) -> None:
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length.")

    ensure_dir(Path(path).parent)
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for idx, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, n, idx)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def list_images(folder: str | Path, extensions: Iterable[str] = (".png", ".jpg", ".jpeg")) -> list[Path]:
    folder = Path(folder)
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(paths))
