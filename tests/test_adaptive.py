from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pi_p.adaptive import (
    MASK_ON,
    AdaptiveParams,
    adaptive_predict,
    adaptive_trace,
    boxes_to_mask,
    boxes_to_windows,
    dilate_mask,
    keep_confident,
    load_params,
    mask_to_rects,
    rects_to_windows,
    window_stages,
)

W, H = 20, 10
IMG_W, IMG_H = 2000, 1200
PARAMS = AdaptiveParams(
    mask_size=500, dilate_kernel=5, margin=0, min_size=0,
    prelim_imgsz=640, prelim_conf=0.25, window_imgsz=640, conf=0.25, iou=0.5,
)
ADAPTIVE_PARAMS = AdaptiveParams(
    mask_size=500, dilate_kernel=5, margin=0, min_size=640,
    prelim_imgsz=1280, prelim_conf=0.1, window_imgsz=640, conf=0.25, iou=0.5,
)
PRELIM = [[400, 200, 720, 520, 0.9, 3], [736, 200, 1056, 520, 0.8, 3]]


def blank_mask():
    return np.zeros((H, W), dtype=np.uint8)


def windows_for(rect, scale, margin, min_size, width=IMG_W, height=IMG_H):
    return rects_to_windows(np.array([rect], dtype=np.float32), scale, margin, min_size, width, height).tolist()


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


def test_mask_to_rects_empty_mask_gives_no_rects():
    assert mask_to_rects(blank_mask()).shape == (0, 4)


def test_mask_to_rects_single_blob_is_its_bounding_box():
    mask = blank_mask()
    mask[3:6, 4:9] = MASK_ON
    assert mask_to_rects(mask).tolist() == [[4, 3, 9, 6]]


def test_mask_to_rects_separate_blobs_give_one_rect_each():
    mask = blank_mask()
    mask[1:3, 1:4] = MASK_ON
    mask[6:9, 12:18] = MASK_ON
    assert sorted(mask_to_rects(mask).tolist()) == [[1, 1, 4, 3], [12, 6, 18, 9]]


def test_mask_to_rects_blobs_touching_at_corner_are_one():
    mask = blank_mask()
    mask[2:4, 2:4] = MASK_ON
    mask[4:6, 4:6] = MASK_ON
    assert mask_to_rects(mask).tolist() == [[2, 2, 6, 6]]


def test_mask_to_rects_l_shape_box_includes_the_gap():
    mask = blank_mask()
    mask[2:8, 3:5] = MASK_ON
    mask[6:8, 3:10] = MASK_ON
    assert mask_to_rects(mask).tolist() == [[3, 2, 10, 8]]


@pytest.mark.parametrize("boxes, expected", [
    ([[2, 4, 6, 6], [10, 4, 14, 6]], [[0, 2, 16, 8]]),
    ([[2, 4, 6, 6], [11, 4, 15, 6]], [[0, 2, 8, 8], [9, 2, 17, 8]]),
])
def test_boxes_dilate_rects_pipeline_merges_or_separates_by_gap(boxes, expected):
    mask = dilate_mask(boxes_to_mask(np.array(boxes, dtype=np.float32), W, H), 5)
    assert sorted(mask_to_rects(mask).tolist()) == expected


def test_rects_to_windows_scales_rect_to_image_pixels():
    assert windows_for([100, 50, 180, 130], 4, 0, 0) == [[400, 200, 720, 520]]


def test_rects_to_windows_adds_margin_on_every_side():
    assert windows_for([100, 50, 180, 130], 4, 10, 0) == [[390, 190, 730, 530]]


def test_rects_to_windows_grows_small_rect_to_min_size_around_center():
    assert windows_for([300, 150, 320, 160], 4, 0, 640) == [[920, 300, 1560, 940]]


def test_rects_to_windows_keeps_rect_already_larger_than_min_size():
    assert windows_for([100, 50, 300, 250], 4, 0, 640) == [[400, 200, 1200, 1000]]


@pytest.mark.parametrize("rect, expected", [
    ([0, 0, 20, 10], [[0, 0, 640, 640]]),
    ([490, 290, 500, 300], [[1360, 560, 2000, 1200]]),
])
def test_rects_to_windows_shifts_rect_back_inside_image(rect, expected):
    assert windows_for(rect, 4, 0, 640) == expected


def test_rects_to_windows_image_smaller_than_min_size_gives_whole_image():
    assert windows_for([10, 10, 30, 20], 1, 0, 640, width=500, height=300) == [[0, 0, 500, 300]]


def test_rects_to_windows_rounds_outward():
    assert windows_for([2, 3, 5, 6], 1.5, 0, 0) == [[3, 4, 8, 9]]


def test_rects_to_windows_no_rects_gives_empty():
    assert rects_to_windows(np.zeros((0, 4)), 4, 0, 640, IMG_W, IMG_H).shape == (0, 4)


def windows_from(boxes, params=PARAMS):
    return boxes_to_windows(np.array(boxes, dtype=np.float32), IMG_W, IMG_H, params).tolist()


def test_boxes_to_windows_no_boxes_gives_empty():
    assert boxes_to_windows(np.zeros((0, 4)), IMG_W, IMG_H, PARAMS).shape == (0, 4)


def test_boxes_to_windows_single_box_grows_by_kernel_radius():
    assert windows_from([[400, 200, 720, 520]]) == [[392, 192, 728, 528]]


def test_boxes_to_windows_merges_boxes_with_small_gap():
    assert windows_from([[400, 200, 720, 520], [736, 200, 1056, 520]]) == [[392, 192, 1064, 528]]


def test_boxes_to_windows_keeps_distant_boxes_separate():
    out = windows_from([[400, 200, 720, 520], [1200, 200, 1520, 520]])
    assert sorted(out) == [[392, 192, 728, 528], [1192, 192, 1528, 528]]


def test_boxes_to_windows_small_box_grows_to_min_size_inside_image():
    assert windows_from([[400, 200, 480, 240]], replace(PARAMS, min_size=640)) == [[120, 0, 760, 640]]


def test_load_params_reads_all_nine_keys(tmp_path):
    path = tmp_path / "adaptive.yaml"
    path.write_text(
        "mask_size: 300\ndilate_kernel: 7\nmargin: 12.5\nmin_size: 320\n"
        "prelim_imgsz: 1280\nprelim_conf: 0.1\nwindow_imgsz: 960\nconf: 0.3\niou: 0.6\n"
    )
    assert load_params(str(path)) == AdaptiveParams(
        mask_size=300, dilate_kernel=7, margin=12.5, min_size=320,
        prelim_imgsz=1280, prelim_conf=0.1, window_imgsz=960, conf=0.3, iou=0.6,
    )


def test_load_params_reads_project_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "adaptive.yaml"
    assert isinstance(load_params(str(path)), AdaptiveParams)


class StubModel:
    def __init__(self, prelim, per_window):
        self.outputs = [[prelim], per_window]
        self.calls = []

    def predict(self, source, imgsz, conf, device, verbose):
        self.calls.append((imgsz, conf))
        return [
            SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(d, dtype=torch.float32).reshape(len(d), 6)))
            for d in self.outputs[len(self.calls) - 1]
        ]


def run_adaptive(prelim, per_window):
    model = StubModel(prelim, per_window)
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    det, windows = adaptive_predict(model, img, ADAPTIVE_PARAMS, "cpu")
    return model, det, windows


def test_adaptive_predict_no_prelim_detections_skips_window_call():
    model, det, windows = run_adaptive([], [])
    assert windows.shape == (0, 4)
    assert det.shape == (0, 6)
    assert len(model.calls) == 1


def test_adaptive_predict_moves_window_detections_to_image_coordinates():
    model, det, windows = run_adaptive(PRELIM, [[[100, 50, 150, 100, 0.7, 0]]])
    assert windows.tolist() == [[392, 40, 1064, 680]]
    assert len(det) == 3
    assert torch.allclose(det[det[:, 5] == 0], torch.tensor([[492, 90, 542, 140, 0.7, 0]]))
    assert model.calls == [(1280, 0.1), (640, 0.25)]


def test_adaptive_predict_merge_suppresses_window_duplicate_of_prelim():
    _, det, _ = run_adaptive(PRELIM, [[[8, 160, 328, 480, 0.6, 3]]])
    assert len(det) == 2
    assert torch.allclose(det[:, 4], torch.tensor([0.9, 0.8]))


def test_adaptive_predict_drops_window_detection_cut_by_window_edge():
    _, det, _ = run_adaptive(PRELIM, [[[0, 100, 40, 150, 0.7, 0]]])
    assert len(det) == 2


def test_adaptive_predict_prelim_below_conf_builds_window_but_is_not_returned():
    _, det, windows = run_adaptive([[400, 200, 720, 520, 0.2, 3]], [[]])
    assert windows.tolist() == [[240, 40, 880, 680]]
    assert det.shape == (0, 6)


def blank_image():
    return np.zeros((IMG_H, IMG_W, 3), np.uint8)


def test_window_stages_windows_match_boxes_to_windows():
    boxes = np.array([[400, 200, 720, 520], [736, 200, 1056, 520]], dtype=np.float32)
    stages = window_stages(boxes, IMG_W, IMG_H, PARAMS)
    np.testing.assert_array_equal(stages.windows, boxes_to_windows(boxes, IMG_W, IMG_H, PARAMS))
    assert stages.scale == 4.0
    assert stages.rects.tolist() == [[98, 48, 266, 132]]


@pytest.mark.parametrize("prelim, per_window", [
    (PRELIM, [[[100, 50, 150, 100, 0.7, 0]]]),
    ([[400, 200, 720, 520, 0.2, 3]], [[]]),
])
def test_adaptive_trace_matches_adaptive_predict(prelim, per_window):
    det, windows = adaptive_predict(StubModel(prelim, per_window), blank_image(), ADAPTIVE_PARAMS, "cpu")
    trace = adaptive_trace(StubModel(prelim, per_window), blank_image(), ADAPTIVE_PARAMS, "cpu")
    assert torch.equal(trace.detections, det)
    np.testing.assert_array_equal(trace.stages.windows, windows)
    assert len(trace.prelim) == len(prelim)


def test_adaptive_trace_without_prelim_detections_has_empty_stages():
    model = StubModel([], [])
    trace = adaptive_trace(model, blank_image(), ADAPTIVE_PARAMS, "cpu")
    assert trace.prelim.shape == (0, 6)
    assert trace.detections.shape == (0, 6)
    assert trace.stages.rects.shape == (0, 4)
    assert trace.stages.windows.shape == (0, 4)
    assert trace.stages.scale == 4.0
    assert trace.stages.mask.shape == trace.stages.dilated.shape == (300, 500)
    assert not trace.stages.mask.any()
    assert not trace.stages.dilated.any()
    assert len(model.calls) == 1


def test_keep_confident_keeps_scores_at_or_above_threshold():
    det = torch.tensor([
        [0, 0, 10, 10, 0.9, 0],
        [0, 0, 10, 10, 0.25, 1],
        [0, 0, 10, 10, 0.1, 2],
        [0, 0, 10, 10, 0.6, 3],
    ])
    assert torch.equal(keep_confident(det, 0.25), det[[0, 1, 3]])


def test_keep_confident_empty_gives_empty():
    assert keep_confident(torch.zeros((0, 6)), 0.25).shape == (0, 6)
