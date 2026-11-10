import numpy as np

from pi_p.adaptive import MASK_ON, boxes_to_mask

W, H = 20, 10


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
