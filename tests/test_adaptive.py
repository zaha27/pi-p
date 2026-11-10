import numpy as np
import pytest

from pi_p.adaptive import MASK_ON, boxes_to_mask, dilate_mask

W, H = 20, 10


def blank_mask():
    return np.zeros((H, W), dtype=np.uint8)


def test_boxes_to_mask_no_boxes_gives_empty_mask():
    mask = boxes_to_mask(np.zeros((0, 4), dtype=np.float32), W, H)
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8
    assert not mask.any()


def test_boxes_to_mask_integer_box_fills_exact_pixels():
    mask = boxes_to_mask(np.array([[2, 3, 6, 5]], dtype=np.float32), W, H)
    assert np.count_nonzero(mask == MASK_ON) == 8
    assert (mask[3:5, 2:6] == MASK_ON).all()


def test_boxes_to_mask_fractional_box_rounds_outward():
    mask = boxes_to_mask(np.array([[2.3, 3.6, 5.2, 4.1]], dtype=np.float32), W, H)
    assert np.count_nonzero(mask == MASK_ON) == 8
    assert (mask[3:5, 2:6] == MASK_ON).all()


def test_boxes_to_mask_overlapping_boxes_are_not_double_counted():
    mask = boxes_to_mask(np.array([[0, 0, 4, 4], [2, 2, 6, 6]], dtype=np.float32), W, H)
    assert np.count_nonzero(mask == MASK_ON) == 28
    assert set(np.unique(mask)) == {0, MASK_ON}


def test_boxes_to_mask_clips_box_partly_outside():
    mask = boxes_to_mask(np.array([[-3, -2, 4, 3]], dtype=np.float32), W, H)
    assert np.count_nonzero(mask == MASK_ON) == 12
    assert (mask[0:3, 0:4] == MASK_ON).all()


def test_boxes_to_mask_ignores_box_fully_outside():
    mask = boxes_to_mask(np.array([[30, 30, 40, 40], [-10, -10, -2, -2]], dtype=np.float32), W, H)
    assert not mask.any()


def test_dilate_mask_single_pixel_becomes_square():
    mask = blank_mask()
    mask[5, 10] = MASK_ON
    out = dilate_mask(mask, 3)
    assert np.count_nonzero(out == MASK_ON) == 9
    assert (out[4:7, 9:12] == MASK_ON).all()


def test_dilate_mask_grows_rectangle_on_every_side():
    mask = blank_mask()
    mask[3:5, 8:12] = MASK_ON
    out = dilate_mask(mask, 5)
    assert np.count_nonzero(out == MASK_ON) == 48
    assert (out[1:7, 6:14] == MASK_ON).all()


def test_dilate_mask_merges_boxes_with_small_gap():
    mask = blank_mask()
    mask[4:6, 2:6] = MASK_ON
    mask[4:6, 10:14] = MASK_ON
    out = dilate_mask(mask, 5)
    assert (out[4, 6:10] == MASK_ON).all()


def test_dilate_mask_keeps_boxes_with_big_gap_apart():
    mask = blank_mask()
    mask[4:6, 2:6] = MASK_ON
    mask[4:6, 11:15] = MASK_ON
    out = dilate_mask(mask, 5)
    assert out[4, 8] == 0
    assert out[4, 7] == MASK_ON
    assert out[4, 9] == MASK_ON


def test_dilate_mask_clips_growth_at_image_corner():
    mask = blank_mask()
    mask[0, 0] = MASK_ON
    out = dilate_mask(mask, 3)
    assert np.count_nonzero(out == MASK_ON) == 4
    assert (out[0:2, 0:2] == MASK_ON).all()


def test_dilate_mask_does_not_modify_input_and_keeps_type():
    mask = blank_mask()
    mask[5, 10] = MASK_ON
    before = mask.copy()
    out = dilate_mask(mask, 3)
    np.testing.assert_array_equal(mask, before)
    assert out.dtype == np.uint8
    assert out.shape == (H, W)


@pytest.mark.parametrize("kernel_size", [4, 0, -3])
def test_dilate_mask_rejects_even_or_non_positive_kernel(kernel_size):
    with pytest.raises(ValueError):
        dilate_mask(blank_mask(), kernel_size)
