import numpy as np

from pi_p.adaptive import adaptive_windows, boxes_to_mask, cover_group, group_boxes


def test_nearby_boxes_form_one_group_and_far_ones_another():
    boxes = np.array([[100, 100, 120, 120], [140, 100, 160, 120], [1500, 800, 1520, 820]], dtype=np.float32)
    mask = boxes_to_mask(boxes, 1920, 1080)
    assert len(group_boxes(mask, 0)) == 3
    groups = group_boxes(mask, 16)
    assert len(groups) == 2
    assert groups[0][0] <= 100 and groups[0][2] >= 160


def test_small_group_gets_one_full_size_window_inside_image():
    wins = cover_group((10, 10, 60, 60), 1920, 1080, 640, 0.2)
    assert wins == [(0, 0, 640, 640)]
    wins = cover_group((1000, 500, 1100, 560), 1920, 1080, 640, 0.2)
    assert len(wins) == 1
    x1, y1, x2, y2 = wins[0]
    assert (x2 - x1, y2 - y1) == (640, 640)
    assert x1 <= 1000 and x2 >= 1100 and y1 <= 500 and y2 >= 560


def test_large_group_is_covered_by_several_windows():
    group = (100, 100, 1500, 700)
    wins = cover_group(group, 1920, 1080, 640, 0.2)
    assert len(wins) > 1
    covered = np.zeros((1080, 1920), dtype=bool)
    for x1, y1, x2, y2 in wins:
        assert (x2 - x1, y2 - y1) == (640, 640)
        covered[y1:y2, x1:x2] = True
    assert covered[100:700, 100:1500].all()


def test_adaptive_windows_cover_every_box_and_skip_empty_areas():
    boxes = np.array([[100, 100, 120, 120], [140, 100, 160, 120], [1500, 800, 1520, 820]], dtype=np.float32)
    wins = adaptive_windows(boxes, 1920, 1080, 640, 0.2, 32)
    assert len(wins) == 2
    for bx1, by1, bx2, by2 in boxes:
        assert any(x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2 for x1, y1, x2, y2 in wins)
    assert not any(x1 <= 960 < x2 and y1 <= 100 < y2 for x1, y1, x2, y2 in wins)


def test_no_boxes_means_no_windows_and_small_image_is_one_window():
    assert adaptive_windows(np.zeros((0, 4), dtype=np.float32), 1920, 1080, 640, 0.2, 32) == []
    assert adaptive_windows(np.array([[10, 10, 30, 30]], dtype=np.float32), 300, 200, 640, 0.2, 32) == [(0, 0, 300, 200)]
