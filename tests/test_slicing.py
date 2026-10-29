import numpy as np

from pi_p.slicing import crop_boxes, windows


def test_windows_cover_image_and_align_to_edges():
    wins = windows(1360, 765, 640, 0.2)
    assert sorted({x for x, *_ in wins}) == [0, 512, 720]
    assert sorted({y for _, y, *_ in wins}) == [0, 125]
    assert all(x2 - x1 == 640 and y2 - y1 == 640 for x1, y1, x2, y2 in wins)


def test_windows_small_image_is_single_window():
    assert windows(300, 200, 640, 0.2) == [(0, 0, 300, 200)]


def test_crop_boxes_visibility_and_shift():
    xyxy = np.array([[90, 10, 190, 20], [170, 10, 270, 20], [300, 300, 310, 310]], dtype=np.float32)
    keep, boxes = crop_boxes(xyxy, (100, 0, 200, 100), 0.5)
    assert keep.tolist() == [True, False, False]
    np.testing.assert_allclose(boxes, [[0, 10, 90, 20]])


def test_merge_drops_fragments_inside_larger_box():
    import torch

    from pi_p.predict import merge

    det = torch.tensor([
        [0, 0, 100, 50, 0.9, 5],
        [0, 0, 40, 50, 0.6, 5],
        [10, 10, 30, 30, 0.8, 0],
        [200, 200, 220, 220, 0.7, 5],
    ])
    assert merge(det, 0.5)[:, 4].tolist() == [0.8999999761581421, 0.800000011920929, 0.699999988079071]


def test_drop_cut_keeps_boxes_on_image_border_only():
    import torch

    from pi_p.predict import drop_cut

    det = torch.tensor([
        [0, 10, 20, 30, 0.9, 0],
        [300, 10, 320, 30, 0.9, 0],
        [100, 100, 120, 120, 0.9, 0],
    ])
    assert drop_cut(det, (0, 0, 320, 320), 1000, 1000)[:, 0].tolist() == [0, 100]
