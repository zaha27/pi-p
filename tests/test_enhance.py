import numpy as np

from pi_p.enhance import clahe, needs_clahe
from pi_p.stats import brightness_contrast, group


def test_black_image_is_dark():
    b, c = brightness_contrast(np.zeros((64, 64, 3), dtype=np.uint8))
    assert group(b, c) == "dark"


def test_flat_gray_image_is_low_contrast():
    b, c = brightness_contrast(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert group(b, c) == "low_contrast"


def test_half_black_half_white_is_normal():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 32:] = 255
    assert group(*brightness_contrast(img)) == "normal"


def test_clahe_keeps_shape_and_raises_contrast():
    rng = np.random.default_rng(0)
    img = rng.integers(110, 140, (128, 128, 3), dtype=np.uint8)
    out = clahe(img)
    assert out.shape == img.shape
    assert brightness_contrast(out)[1] > brightness_contrast(img)[1]


def test_adaptive_skips_normal_images():
    groups = {"a.jpg": "dark", "b.jpg": "normal", "c.jpg": "low_contrast"}
    assert [needs_clahe(n, "adaptive", groups) for n in groups] == [True, False, True]
    assert all(needs_clahe(n, "uniform", groups) for n in groups)
