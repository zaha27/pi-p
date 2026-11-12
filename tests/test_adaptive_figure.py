from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from pi_p.adaptive import AdaptiveParams, adaptive_trace
from pi_p.adaptive_figure import build_comparison_figure, build_figure, save_figure

IMG_W, IMG_H = 2000, 1200
PARAMS = AdaptiveParams(
    mask_size=500, dilate_kernel=5, margin=0, min_size=640,
    prelim_imgsz=640, prelim_conf=0.25, window_imgsz=640, conf=0.25, iou=0.5,
)
BOXES = [[400, 200, 720, 520, 0.9, 3]]


class FakeModel:
    def __init__(self, boxes):
        self.data = torch.tensor(boxes, dtype=torch.float32).reshape(len(boxes), 6)

    def predict(self, source, imgsz, conf, device, verbose):
        return [SimpleNamespace(boxes=SimpleNamespace(data=self.data)) for _ in source]


def blank_image():
    return np.zeros((IMG_H, IMG_W, 3), np.uint8)


def make_trace(boxes):
    return adaptive_trace(FakeModel(boxes), blank_image(), PARAMS, "cpu")


@pytest.mark.parametrize("boxes", [BOXES, []], ids=["with_boxes", "no_boxes"])
def test_build_figure_has_six_axes(boxes):
    fig = build_figure(blank_image(), make_trace(boxes))
    assert len(fig.axes) == 6
    plt.close(fig)


def test_save_figure_writes_non_empty_png(tmp_path):
    out = tmp_path / "sub" / "fig.png"
    save_figure(build_figure(blank_image(), make_trace(BOXES)), out)
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("boxes, n_det, n_win", [(BOXES, 1, 1), ([], 0, 0)], ids=["with_boxes", "no_boxes"])
def test_build_comparison_figure_has_two_axes_and_reports_counts(boxes, n_det, n_win):
    fig = build_comparison_figure(blank_image(), make_trace(boxes), PARAMS)
    left, right = fig.axes
    assert f"{n_det} detectii" in left.get_title()
    assert f"{n_det} detectii" in right.get_title()
    assert f"{n_win} ferestre" in right.get_title()
    plt.close(fig)


def test_build_figure_first_panels_use_seed_boxes_only():
    trace = adaptive_trace(FakeModel([[400, 200, 720, 520, 0.9, 3], [1500, 800, 1560, 880, 0.1, 0]]), blank_image(), replace(PARAMS, conf=0.05), "cpu")
    fig = build_figure(blank_image(), trace)
    assert "1 casete" in fig.axes[0].get_title()
    assert "1 casete" in fig.axes[1].get_title()
    plt.close(fig)
