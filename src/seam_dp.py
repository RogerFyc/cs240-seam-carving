"""Dynamic programming seam search and seam removal."""

from __future__ import annotations

import numpy as np


def find_vertical_seam_dp(energy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if energy.ndim != 2:
        raise ValueError("energy must be a 2D array.")

    h, w = energy.shape
    dp = np.zeros((h, w), dtype=np.float64)
    parent = np.zeros((h, w), dtype=np.int32)

    dp[0] = energy[0]
    parent[0] = -1

    for i in range(1, h):
        for j in range(w):
            candidates = []
            for prev_j in (j - 1, j, j + 1):
                if 0 <= prev_j < w:
                    candidates.append((dp[i - 1, prev_j], prev_j))
            best_cost, best_prev_j = min(candidates, key=lambda x: x[0])
            dp[i, j] = energy[i, j] + best_cost
            parent[i, j] = best_prev_j

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for i in range(h - 1, 0, -1):
        seam[i - 1] = parent[i, seam[i]]

    cost = float(dp[-1, seam[-1]])
    return seam, cost, dp


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if len(seam) != h:
        raise ValueError(f"Seam length {len(seam)} does not match image height {h}.")
    if w <= 1:
        raise ValueError("Cannot remove seam from image with width <= 1.")

    if image.ndim == 2:
        out = np.zeros((h, w - 1), dtype=image.dtype)
        for i in range(h):
            j = int(seam[i])
            out[i] = np.concatenate([image[i, :j], image[i, j + 1:]])
        return out

    channels = image.shape[2]
    out = np.zeros((h, w - 1, channels), dtype=image.dtype)
    for i in range(h):
        j = int(seam[i])
        out[i] = np.concatenate([image[i, :j], image[i, j + 1:]], axis=0)
    return out


def find_horizontal_seam_dp(energy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    seam_t, cost, dp_t = find_vertical_seam_dp(energy.T)
    return seam_t, cost, dp_t.T


def remove_horizontal_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        transposed = image.T
        removed = remove_vertical_seam(transposed, seam)
        return removed.T
    transposed = np.transpose(image, (1, 0, 2))
    removed = remove_vertical_seam(transposed, seam)
    return np.transpose(removed, (1, 0, 2))
