from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pi_p.adaptive import AdaptiveParams, boxes_to_windows, load_params
from pi_p.window_merge import merge_windows

IMG_W, IMG_H = 2000, 1200
PARAMS = AdaptiveParams(
    mask_size=500, dilate_kernel=5, margin=0, min_size=640,
    prelim_imgsz=640, prelim_conf=0.25, window_imgsz=640, conf=0.25, iou=0.5,
)
YAML_NINE_KEYS = "mask_size: 300\ndilate_kernel: 7\nmargin: 12.5\nmin_size: 320\nprelim_imgsz: 1280\nprelim_conf: 0.1\nwindow_imgsz: 960\nconf: 0.3\niou: 0.6\n"
CHAIN = [[0, 0, 100, 100], [50, 0, 150, 100], [100, 0, 200, 100], [500, 500, 600, 600], [550, 520, 650, 620]]


def wins(*rows):
    return np.array(rows, dtype=np.int32)


def test_merge_windows_overlapping_windows_become_their_union():
    out = merge_windows(wins([0, 0, 100, 100], [20, 0, 120, 100]), 0.5, 200)
    assert out.tolist() == [[0, 0, 120, 100]]


def test_merge_windows_disjoint_windows_stay_unchanged():
    out = merge_windows(wins([0, 0, 100, 100], [300, 300, 400, 400]), 0.5, 1000)
    assert out.tolist() == [[0, 0, 100, 100], [300, 300, 400, 400]]


def test_merge_windows_respects_max_window():
    out = merge_windows(wins([0, 0, 100, 100], [20, 0, 120, 100], [100, 0, 200, 100]), 0.1, 150)
    assert out.tolist() == [[0, 0, 120, 100], [100, 0, 200, 100]]
    assert ((out[:, 2] - out[:, 0]) <= 150).all()


def test_merge_windows_covers_every_original_window():
    original = wins(*CHAIN)
    out = merge_windows(original, 0.3, 1000)
    assert len(out) < len(original)
    for x1, y1, x2, y2 in original:
        assert ((out[:, 0] <= x1) & (out[:, 1] <= y1) & (out[:, 2] >= x2) & (out[:, 3] >= y2)).any()


def test_merge_windows_second_run_gives_same_result():
    once = merge_windows(wins(*CHAIN), 0.3, 1000)
    np.testing.assert_array_equal(merge_windows(once, 0.3, 1000), once)


def test_merge_windows_overlap_zero_returns_input_unchanged():
    original = wins([0, 0, 100, 100], [20, 0, 120, 100])
    np.testing.assert_array_equal(merge_windows(original, 0, 1000), original)


@pytest.mark.parametrize("rows", [[], [[0, 0, 10, 10]]], ids=["no_windows", "single_window"])
def test_merge_windows_trivial_inputs_stay_unchanged(rows):
    original = np.array(rows, dtype=np.int32).reshape(-1, 4)
    out = merge_windows(original, 0.5, 1000)
    np.testing.assert_array_equal(out, original)
    assert out.dtype == np.int32


def test_merge_windows_threshold_is_strict():
    pair = wins([0, 0, 100, 100], [50, 0, 150, 100])
    assert len(merge_windows(pair, 0.5, 1000)) == 2
    assert len(merge_windows(pair, 0.49, 1000)) == 1


def test_merge_windows_does_not_modify_input_and_keeps_int32():
    original = wins([0, 0, 100, 100], [20, 0, 120, 100])
    before = original.copy()
    out = merge_windows(original, 0.5, 1000)
    np.testing.assert_array_equal(original, before)
    assert out.dtype == np.int32


def test_boxes_to_windows_merges_overlapping_windows_only_when_enabled():
    boxes = np.array([[400, 200, 720, 520], [800, 200, 1120, 520]], dtype=np.float32)
    separate = [[240, 40, 880, 680], [640, 40, 1280, 680]]
    assert boxes_to_windows(boxes, IMG_W, IMG_H, PARAMS).tolist() == separate
    merged = boxes_to_windows(boxes, IMG_W, IMG_H, replace(PARAMS, merge_overlap=0.3, max_window=1280))
    assert merged.tolist() == [[240, 40, 1280, 680]]


def test_load_params_without_merge_keys_uses_disabled_defaults(tmp_path):
    path = tmp_path / "adaptive.yaml"
    path.write_text(YAML_NINE_KEYS)
    params = load_params(str(path))
    assert params.merge_overlap == 0
    assert params.max_window == 1280


def test_load_params_reads_merge_keys_and_accepts_override(tmp_path):
    path = tmp_path / "adaptive.yaml"
    path.write_text(YAML_NINE_KEYS + "merge_overlap: 0.2\nmax_window: 960\n")
    assert (load_params(str(path)).merge_overlap, load_params(str(path)).max_window) == (0.2, 960)
    overridden = load_params(str(path), ("merge_overlap=0.5",))
    assert (overridden.merge_overlap, overridden.max_window) == (0.5, 960)


def test_project_config_keeps_merge_disabled():
    path = Path(__file__).resolve().parents[1] / "configs" / "adaptive.yaml"
    assert load_params(str(path)).merge_overlap == 0
