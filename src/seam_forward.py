"""Forward-energy seam search.

Forward energy estimates the visual discontinuity introduced after removing
a seam. Compared with backward energy, it often produces more stable visual
results because it considers the new neighboring pixels created by deletion.
"""

from __future__ import annotations

import numpy as np

from .energy import gradient_energy, rgb_to_grayscale


def _transpose_image(image: np.ndarray) -> np.ndarray:
    """Transpose an image by swapping height and width."""
    if image.ndim == 2:
        return image.T
    return np.transpose(image, (1, 0, 2))


def find_vertical_seam_forward(
    image: np.ndarray,
    pixel_penalty: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find a vertical seam using forward-energy dynamic programming.

    Args:
        image: RGB image of shape H x W x 3, or grayscale image H x W.
        pixel_penalty: optional H x W matrix. Positive values discourage
            the seam from passing through a pixel, while negative values
            encourage it. This is used for protection/removal masks.

    Returns:
        seam: array of length H, where seam[i] is the chosen column at row i.
        cost: total forward-energy DP cost.
        dp: cumulative DP table.
    """
    gray = rgb_to_grayscale(image)
    h, w = gray.shape

    if w <= 0 or h <= 0:
        raise ValueError("image must have positive height and width.")

    base_energy = gradient_energy(image)

    if pixel_penalty is not None:
        if pixel_penalty.shape != (h, w):
            raise ValueError(
                f"pixel_penalty shape {pixel_penalty.shape} does not match image shape {(h, w)}."
            )
        base_energy = base_energy + pixel_penalty

    dp = np.zeros((h, w), dtype=np.float64)
    parent = np.zeros((h, w), dtype=np.int32)

    # The first row has no previous row, so we initialize by pixel energy.
    dp[0] = base_energy[0]
    parent[0] = -1

    for i in range(1, h):
        for j in range(w):
            left = gray[i, j - 1] if j - 1 >= 0 else gray[i, j]
            right = gray[i, j + 1] if j + 1 < w else gray[i, j]
            up = gray[i - 1, j]

            c_u = abs(right - left)
            c_l = c_u + abs(up - left)
            c_r = c_u + abs(up - right)

            candidates: list[tuple[float, int]] = []

            if j - 1 >= 0:
                candidates.append((dp[i - 1, j - 1] + c_l, j - 1))

            candidates.append((dp[i - 1, j] + c_u, j))

            if j + 1 < w:
                candidates.append((dp[i - 1, j + 1] + c_r, j + 1))

            best_cost, best_prev_j = min(candidates, key=lambda x: x[0])

            dp[i, j] = base_energy[i, j] + best_cost
            parent[i, j] = best_prev_j

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))

    for i in range(h - 1, 0, -1):
        seam[i - 1] = parent[i, seam[i]]

    cost = float(dp[-1, seam[-1]])
    return seam, cost, dp


def find_horizontal_seam_forward(
    image: np.ndarray,
    pixel_penalty: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find a horizontal seam using forward energy.

    This is implemented by transposing the image and then applying the
    vertical forward-energy seam algorithm.
    """
    image_t = _transpose_image(image)
    penalty_t = pixel_penalty.T if pixel_penalty is not None else None
    seam_t, cost, dp_t = find_vertical_seam_forward(image_t, penalty_t)

    # seam_t has length W and gives row indices in the original image.
    return seam_t, cost, dp_t.T