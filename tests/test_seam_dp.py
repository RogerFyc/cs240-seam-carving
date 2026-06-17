import numpy as np

from src.seam_dp import find_vertical_seam_dp, remove_vertical_seam


def test_find_vertical_seam_dp_simple():
    energy = np.array([
        [5, 1, 5],
        [5, 1, 5],
        [5, 1, 5],
    ], dtype=float)
    seam, cost, dp = find_vertical_seam_dp(energy)
    assert seam.tolist() == [1, 1, 1]
    assert cost == 3


def test_remove_vertical_seam_shape():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    seam = np.array([2, 2, 2, 2])
    out = remove_vertical_seam(image, seam)
    assert out.shape == (4, 4, 3)
