"""Standard image resizing baseline."""

from __future__ import annotations

import numpy as np
from PIL import Image


def standard_resize(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    if target_width is None and target_height is None:
        raise ValueError("At least one of target_width or target_height must be provided.")

    if target_width is None:
        scale = target_height / h
        target_width = max(1, int(round(w * scale)))
    if target_height is None:
        scale = target_width / w
        target_height = max(1, int(round(h * scale)))

    pil_img = Image.fromarray(image.astype(np.uint8))
    resized = pil_img.resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
