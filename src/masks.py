"""Mask utilities for object protection and object removal.

A protection mask increases pixel energy so seams avoid the region.
A removal mask decreases pixel energy so seams prefer to pass through it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_rect_mask(
    image_shape: tuple[int, ...],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Create a rectangular boolean mask.

    Coordinates follow image convention:
        x = column index,
        y = row index.

    Args:
        image_shape: image shape, usually H x W x 3 or H x W.
        x0, y0, x1, y1: rectangle coordinates. The interval is
            [x0, x1) in x and [y0, y1) in y.

    Returns:
        Boolean mask of shape H x W.
    """
    h, w = image_shape[:2]

    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def load_mask(mask_path: str | Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a mask image and convert it to a boolean H x W mask.

    White / bright pixels are treated as True.
    """
    h, w = target_shape
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((w, h), Image.Resampling.NEAREST)
    arr = np.asarray(mask_img)
    return arr > 127


def make_mask_penalty(
    shape: tuple[int, int],
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> np.ndarray:
    """Create an additive energy penalty matrix from masks.

    Positive penalty discourages seam passing through protected pixels.
    Negative penalty encourages seam passing through removal pixels.
    """
    h, w = shape
    penalty = np.zeros((h, w), dtype=np.float64)

    if protect_mask is not None:
        if protect_mask.shape != (h, w):
            raise ValueError(
                f"protect_mask shape {protect_mask.shape} does not match {(h, w)}."
            )
        penalty[protect_mask.astype(bool)] += protect_weight

    if remove_mask is not None:
        if remove_mask.shape != (h, w):
            raise ValueError(
                f"remove_mask shape {remove_mask.shape} does not match {(h, w)}."
            )
        penalty[remove_mask.astype(bool)] -= remove_weight

    return penalty


def apply_mask_to_energy(
    energy: np.ndarray,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> np.ndarray:
    """Return a copy of energy modified by protection/removal masks."""
    penalty = make_mask_penalty(
        energy.shape,
        protect_mask=protect_mask,
        remove_mask=remove_mask,
        protect_weight=protect_weight,
        remove_weight=remove_weight,
    )
    return energy + penalty


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a boolean mask on an image for visualization."""
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {image.shape[:2]}.")

    out = image.astype(np.float64).copy()
    color_arr = np.array(color, dtype=np.float64)

    mask_bool = mask.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)