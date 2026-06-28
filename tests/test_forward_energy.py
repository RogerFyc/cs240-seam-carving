import numpy as np

from src.seam_forward import find_vertical_seam_forward


def test_forward_energy_output_shape():
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    seam, cost, dp = find_vertical_seam_forward(image)

    assert seam.shape == (5,)
    assert dp.shape == (5, 6)
    assert isinstance(cost, float)


def test_forward_energy_seam_is_connected():
    image = np.random.randint(0, 255, size=(8, 9, 3), dtype=np.uint8)
    seam, cost, dp = find_vertical_seam_forward(image)

    assert len(seam) == image.shape[0]
    assert np.all(seam >= 0)
    assert np.all(seam < image.shape[1])
    assert np.all(np.abs(np.diff(seam)) <= 1)