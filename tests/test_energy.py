import numpy as np

from src.energy import gradient_energy


def test_gradient_energy_shape():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    energy = gradient_energy(image)
    assert energy.shape == (10, 12)


def test_gradient_energy_nonnegative():
    image = np.random.randint(0, 255, size=(8, 9, 3), dtype=np.uint8)
    energy = gradient_energy(image)
    assert np.all(energy >= 0)
