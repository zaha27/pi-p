import numpy as np
import torch
from ultralytics.utils.metrics import box_iou

from pi_p.evaluate import evaluate_image, match, summarize


def test_match_needs_same_class_and_uses_each_object_once():
    gt = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30.0]])
    gt_cls = np.array([0, 1])
    pred = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [21, 21, 30, 30], [20, 20, 30, 30.0]])
    pred_cls = np.array([0, 0, 1, 0])
    correct = match(pred_cls, gt_cls, box_iou(gt, pred).numpy())
    assert correct[:, 0].tolist() == [True, False, True, False]
    assert correct[0].all()
    assert correct[2].tolist() == [True] * 7 + [False] * 3


def test_perfect_detections_give_map_one_and_misses_lower_it():
    labels = np.array([[3, 0.25, 0.25, 0.2, 0.2], [0, 0.75, 0.75, 0.1, 0.1]], dtype=np.float32)
    perfect = torch.tensor([[15, 15, 35, 35, 0.9, 3], [70, 70, 80, 80, 0.8, 0.0]])
    s = summarize([evaluate_image(perfect, labels, 100, 100)])
    assert np.isclose(s["mAP50"], 1, atol=0.01) and np.isclose(s["mAP50-95"], 1, atol=0.01)

    only_car = summarize([evaluate_image(perfect[:1], labels, 100, 100)])
    assert np.isclose(only_car["mAP50"], 0.5, atol=0.01)
    assert only_car["pe_clase"]["pedestrian"]["mAP50"] == 0


def test_image_without_detections_or_labels():
    empty_det = evaluate_image(torch.zeros((0, 6)), np.array([[3, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32), 100, 100)
    assert empty_det["correct"].shape == (0, 10) and empty_det["gt_cls"].tolist() == [3]
    no_labels = evaluate_image(torch.tensor([[0, 0, 10, 10, 0.9, 3.0]]), np.zeros((0, 5), dtype=np.float32), 100, 100)
    assert not no_labels["correct"].any()
    assert summarize([]) is None
