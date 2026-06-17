"""Energy functions for seam carving."""

from __future__ import annotations

import numpy as np

from .utils import image_to_float


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    image_f = image_to_float(image)
    if image_f.ndim == 2:
        return image_f
    if image_f.shape[2] < 3:
        raise ValueError("Expected an RGB image with 3 channels.")
    r = image_f[..., 0]
    g = image_f[..., 1]
    b = image_f[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def gradient_energy(image: np.ndarray) -> np.ndarray:
    """Compute backward energy using image gradients."""
    gray = rgb_to_grayscale(image)

    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)
    up = np.roll(gray, 1, axis=0)
    down = np.roll(gray, -1, axis=0)

    left[:, 0] = gray[:, 0]
    right[:, -1] = gray[:, -1]
    up[0, :] = gray[0, :]
    down[-1, :] = gray[-1, :]

    dx = np.abs(right - left)
    dy = np.abs(down - up)
    return dx + dy


def simple_energy(image: np.ndarray) -> np.ndarray:
    return gradient_energy(image)
