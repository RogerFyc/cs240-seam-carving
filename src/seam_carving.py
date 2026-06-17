"""High-level seam carving functions."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from tqdm import tqdm

from .energy import gradient_energy
from .seam_dp import (
    find_vertical_seam_dp,
    remove_vertical_seam,
    find_horizontal_seam_dp,
    remove_horizontal_seam,
)
from .seam_greedy import find_vertical_seam_greedy, find_horizontal_seam_greedy


@dataclass
class CarvingResult:
    image: np.ndarray
    elapsed_seconds: float
    removed_seams: int
    method: str


def _find_vertical_seam(energy: np.ndarray, method: str):
    if method == "dp":
        seam, cost, _ = find_vertical_seam_dp(energy)
        return seam, cost
    if method == "greedy":
        return find_vertical_seam_greedy(energy)
    raise ValueError(f"Unknown seam method: {method}")


def _find_horizontal_seam(energy: np.ndarray, method: str):
    if method == "dp":
        seam, cost, _ = find_horizontal_seam_dp(energy)
        return seam, cost
    if method == "greedy":
        return find_horizontal_seam_greedy(energy)
    raise ValueError(f"Unknown seam method: {method}")


def carve_width(
    image: np.ndarray,
    target_width: int,
    method: str = "dp",
    show_progress: bool = True,
) -> CarvingResult:
    if method not in {"dp", "greedy"}:
        raise ValueError("method must be 'dp' or 'greedy'.")

    current = image.copy()
    h, w = current.shape[:2]
    if not (1 <= target_width <= w):
        raise ValueError(f"target_width must be in [1, {w}].")

    num_seams = w - target_width
    start = time.perf_counter()

    iterator = range(num_seams)
    if show_progress:
        iterator = tqdm(iterator, desc=f"carve_width_{method}", leave=False)

    for _ in iterator:
        energy = gradient_energy(current)
        seam, _ = _find_vertical_seam(energy, method)
        current = remove_vertical_seam(current, seam)

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, num_seams, method)


def carve_height(
    image: np.ndarray,
    target_height: int,
    method: str = "dp",
    show_progress: bool = True,
) -> CarvingResult:
    if method not in {"dp", "greedy"}:
        raise ValueError("method must be 'dp' or 'greedy'.")

    current = image.copy()
    h, w = current.shape[:2]
    if not (1 <= target_height <= h):
        raise ValueError(f"target_height must be in [1, {h}].")

    num_seams = h - target_height
    start = time.perf_counter()

    iterator = range(num_seams)
    if show_progress:
        iterator = tqdm(iterator, desc=f"carve_height_{method}", leave=False)

    for _ in iterator:
        energy = gradient_energy(current)
        seam, _ = _find_horizontal_seam(energy, method)
        current = remove_horizontal_seam(current, seam)

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, num_seams, method)


def carve_to_size(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    method: str = "dp",
    show_progress: bool = True,
) -> CarvingResult:
    current = image.copy()
    total_removed = 0
    start = time.perf_counter()

    if target_width is not None and target_width < current.shape[1]:
        width_result = carve_width(current, target_width, method, show_progress)
        current = width_result.image
        total_removed += width_result.removed_seams

    if target_height is not None and target_height < current.shape[0]:
        height_result = carve_height(current, target_height, method, show_progress)
        current = height_result.image
        total_removed += height_result.removed_seams

    elapsed = time.perf_counter() - start
    return CarvingResult(current, elapsed, total_removed, method)
