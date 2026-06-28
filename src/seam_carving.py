"""High-level seam carving functions."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from tqdm import tqdm

from .energy import gradient_energy
from .masks import make_mask_penalty
from .seam_dp import (
    find_vertical_seam_dp,
    remove_vertical_seam,
    find_horizontal_seam_dp,
    remove_horizontal_seam,
)
from .seam_greedy import find_vertical_seam_greedy, find_horizontal_seam_greedy
from .seam_forward import find_vertical_seam_forward, find_horizontal_seam_forward


SUPPORTED_METHODS = {"dp", "greedy", "forward"}


@dataclass
class CarvingResult:
    image: np.ndarray
    elapsed_seconds: float
    removed_seams: int
    method: str
    seam_costs: list[float] = field(default_factory=list)


def _remove_mask_vertical(mask: np.ndarray | None, seam: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return remove_vertical_seam(mask.astype(np.uint8), seam).astype(bool)


def _remove_mask_horizontal(mask: np.ndarray | None, seam: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return remove_horizontal_seam(mask.astype(np.uint8), seam).astype(bool)


def _find_vertical_seam(
    image: np.ndarray,
    method: str,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> tuple[np.ndarray, float]:
    """Find a vertical seam using the selected method."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {SUPPORTED_METHODS}")

    h, w = image.shape[:2]
    penalty = make_mask_penalty(
        (h, w),
        protect_mask=protect_mask,
        remove_mask=remove_mask,
        protect_weight=protect_weight,
        remove_weight=remove_weight,
    )

    if method == "dp":
        energy = gradient_energy(image) + penalty
        seam, cost, _ = find_vertical_seam_dp(energy)
        return seam, cost

    if method == "greedy":
        energy = gradient_energy(image) + penalty
        return find_vertical_seam_greedy(energy)

    if method == "forward":
        seam, cost, _ = find_vertical_seam_forward(image, pixel_penalty=penalty)
        return seam, cost

    raise RuntimeError("Unreachable method branch.")


def _find_horizontal_seam(
    image: np.ndarray,
    method: str,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> tuple[np.ndarray, float]:
    """Find a horizontal seam using the selected method."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {SUPPORTED_METHODS}")

    h, w = image.shape[:2]
    penalty = make_mask_penalty(
        (h, w),
        protect_mask=protect_mask,
        remove_mask=remove_mask,
        protect_weight=protect_weight,
        remove_weight=remove_weight,
    )

    if method == "dp":
        energy = gradient_energy(image) + penalty
        seam, cost, _ = find_horizontal_seam_dp(energy)
        return seam, cost

    if method == "greedy":
        energy = gradient_energy(image) + penalty
        return find_horizontal_seam_greedy(energy)

    if method == "forward":
        seam, cost, _ = find_horizontal_seam_forward(image, pixel_penalty=penalty)
        return seam, cost

    raise RuntimeError("Unreachable method branch.")


def carve_width(
    image: np.ndarray,
    target_width: int,
    method: str = "dp",
    show_progress: bool = True,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> CarvingResult:
    """Reduce image width by repeatedly removing vertical seams.

    Args:
        image: input image.
        target_width: target width after seam removal.
        method: one of "dp", "greedy", or "forward".
        protect_mask: optional H x W boolean mask to protect.
        remove_mask: optional H x W boolean mask to remove.
        protect_weight: energy increase for protected pixels.
        remove_weight: energy decrease for removal pixels.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")

    current = image.copy()
    h, w = current.shape[:2]
    if not (1 <= target_width <= w):
        raise ValueError(f"target_width must be in [1, {w}].")

    current_protect = protect_mask.copy() if protect_mask is not None else None
    current_remove = remove_mask.copy() if remove_mask is not None else None

    if current_protect is not None and current_protect.shape != (h, w):
        raise ValueError("protect_mask shape must match image height and width.")
    if current_remove is not None and current_remove.shape != (h, w):
        raise ValueError("remove_mask shape must match image height and width.")

    num_seams = w - target_width
    seam_costs: list[float] = []
    start = time.perf_counter()

    iterator = range(num_seams)
    if show_progress:
        iterator = tqdm(iterator, desc=f"carve_width_{method}", leave=False)

    for _ in iterator:
        seam, cost = _find_vertical_seam(
            current,
            method=method,
            protect_mask=current_protect,
            remove_mask=current_remove,
            protect_weight=protect_weight,
            remove_weight=remove_weight,
        )
        seam_costs.append(cost)

        current = remove_vertical_seam(current, seam)
        current_protect = _remove_mask_vertical(current_protect, seam)
        current_remove = _remove_mask_vertical(current_remove, seam)

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, num_seams, method, seam_costs)


def carve_height(
    image: np.ndarray,
    target_height: int,
    method: str = "dp",
    show_progress: bool = True,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> CarvingResult:
    """Reduce image height by repeatedly removing horizontal seams."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")

    current = image.copy()
    h, w = current.shape[:2]
    if not (1 <= target_height <= h):
        raise ValueError(f"target_height must be in [1, {h}].")

    current_protect = protect_mask.copy() if protect_mask is not None else None
    current_remove = remove_mask.copy() if remove_mask is not None else None

    if current_protect is not None and current_protect.shape != (h, w):
        raise ValueError("protect_mask shape must match image height and width.")
    if current_remove is not None and current_remove.shape != (h, w):
        raise ValueError("remove_mask shape must match image height and width.")

    num_seams = h - target_height
    seam_costs: list[float] = []
    start = time.perf_counter()

    iterator = range(num_seams)
    if show_progress:
        iterator = tqdm(iterator, desc=f"carve_height_{method}", leave=False)

    for _ in iterator:
        seam, cost = _find_horizontal_seam(
            current,
            method=method,
            protect_mask=current_protect,
            remove_mask=current_remove,
            protect_weight=protect_weight,
            remove_weight=remove_weight,
        )
        seam_costs.append(cost)

        current = remove_horizontal_seam(current, seam)
        current_protect = _remove_mask_horizontal(current_protect, seam)
        current_remove = _remove_mask_horizontal(current_remove, seam)

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, num_seams, method, seam_costs)


def carve_to_size(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    method: str = "dp",
    show_progress: bool = True,
    protect_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    protect_weight: float = 1e6,
    remove_weight: float = 1e6,
) -> CarvingResult:
    """Reduce width and/or height using seam carving.

    Width is reduced first, then height. This is intentionally simple and
    sufficient for the course project.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")

    current = image.copy()
    total_removed = 0
    all_costs: list[float] = []
    start = time.perf_counter()

    current_protect = protect_mask.copy() if protect_mask is not None else None
    current_remove = remove_mask.copy() if remove_mask is not None else None

    if target_width is not None and target_width < current.shape[1]:
        width_result = carve_width(
            current,
            target_width,
            method=method,
            show_progress=show_progress,
            protect_mask=current_protect,
            remove_mask=current_remove,
            protect_weight=protect_weight,
            remove_weight=remove_weight,
        )
        current = width_result.image
        total_removed += width_result.removed_seams
        all_costs.extend(width_result.seam_costs)


    if target_height is not None and target_height < current.shape[0]:
        height_result = carve_height(
            current,
            target_height,
            method=method,
            show_progress=show_progress,
            protect_mask=None if total_removed > 0 else current_protect,
            remove_mask=None if total_removed > 0 else current_remove,
            protect_weight=protect_weight,
            remove_weight=remove_weight,
        )
        current = height_result.image
        total_removed += height_result.removed_seams
        all_costs.extend(height_result.seam_costs)

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, total_removed, method, all_costs)