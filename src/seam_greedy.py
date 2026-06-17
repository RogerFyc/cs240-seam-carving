"""Greedy seam search baseline."""

from __future__ import annotations

import numpy as np


def find_vertical_seam_greedy(energy: np.ndarray) -> tuple[np.ndarray, float]:
    if energy.ndim != 2:
        raise ValueError("energy must be a 2D array.")

    h, w = energy.shape
    seam = np.zeros(h, dtype=np.int32)
    seam[0] = int(np.argmin(energy[0]))
    total_cost = float(energy[0, seam[0]])

    for i in range(1, h):
        prev_j = seam[i - 1]
        candidates = []
        for j in (prev_j - 1, prev_j, prev_j + 1):
            if 0 <= j < w:
                candidates.append((energy[i, j], j))
        best_energy, best_j = min(candidates, key=lambda x: x[0])
        seam[i] = best_j
        total_cost += float(best_energy)

    return seam, total_cost


def find_horizontal_seam_greedy(energy: np.ndarray) -> tuple[np.ndarray, float]:
    return find_vertical_seam_greedy(energy.T)
