import numpy as np

from pi_p.metrics import box_iou, match, mean_ap

GT = np.array([[3, 0, 0, 10, 10]], dtype=np.float32)


def test_box_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    np.testing.assert_allclose(box_iou(a, b), [[50 / 150]])


def test_match_strict_threshold_fails_for_shifted_box():
    det = np.array([[1, 0, 11, 10, 0.9, 3]], dtype=np.float32)
    correct = match(det, GT)
    assert correct[0, 0] and not correct[0, -1]


def test_perfect_detection_gives_ap_one():
    det = np.array([[0, 0, 10, 10, 0.9, 3]], dtype=np.float32)
    map50, map50_95, _ = mean_ap([(det, GT)])
    assert map50 > 0.99 and map50_95 > 0.99


def test_wrong_class_gives_ap_zero():
    det = np.array([[0, 0, 10, 10, 0.9, 4]], dtype=np.float32)
    assert mean_ap([(det, GT)])[0] == 0


def test_duplicate_counts_once():
    det = np.array([[0, 0, 10, 10, 0.9, 3], [0, 0, 10, 10, 0.8, 3]], dtype=np.float32)
    assert match(det, GT)[:, 0].tolist() == [True, False]


def test_false_positive_with_higher_score_lowers_ap():
    good = np.array([[0, 0, 10, 10, 0.5, 3]], dtype=np.float32)
    bad = np.array([[50, 50, 60, 60, 0.9, 3]], dtype=np.float32)
    assert mean_ap([(np.vstack([good, bad]), GT)])[0] < mean_ap([(good, GT)])[0]
