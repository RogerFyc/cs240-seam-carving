import numpy as np

from src.masks import create_rect_mask, apply_mask_to_energy, overlay_mask


def test_create_rect_mask():
    mask = create_rect_mask((10, 12, 3), x0=2, y0=3, x1=7, y1=8)

    assert mask.shape == (10, 12)
    assert mask[3:8, 2:7].all()
    assert not mask[:3, :].any()


def test_apply_protect_mask_to_energy():
    energy = np.zeros((5, 5), dtype=float)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    modified = apply_mask_to_energy(energy, protect_mask=mask, protect_weight=100.0)

    assert modified[2, 2] == 100.0
    assert modified[0, 0] == 0.0


def test_apply_remove_mask_to_energy():
    energy = np.zeros((5, 5), dtype=float)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    modified = apply_mask_to_energy(energy, remove_mask=mask, remove_weight=100.0)

    assert modified[2, 2] == -100.0
    assert modified[0, 0] == 0.0


def test_overlay_mask_shape():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True

    out = overlay_mask(image, mask)

    assert out.shape == image.shape