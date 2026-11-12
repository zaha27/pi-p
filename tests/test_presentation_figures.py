import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from pi_p import slicing
from pi_p.adaptive import AdaptiveParams, adaptive_trace
from pi_p.presentation_figures import (
    METHOD_LABELS,
    WINDOW_FILL_ALPHA,
    Setup,
    adaptive_detector,
    build_merge_figure,
    build_methods_figure,
    build_stages_figure,
    figure_paths,
    grid_windows,
    method_title,
    parse_args,
    pixels_processed,
    render_image,
    run_methods,
    sahi_windows,
)

IMG_W, IMG_H = 2000, 1200
PARAMS = AdaptiveParams(
    mask_size=500, dilate_kernel=5, margin=0, min_size=640,
    prelim_imgsz=640, prelim_conf=0.25, window_imgsz=640, conf=0.25, iou=0.5,
)
BOXES = [[400, 200, 720, 520, 0.9, 3]]
OLD_NAMES = {"ferestre_adaptive_pasi.png", "ferestre_adaptive_comparatie.png"}


class FakeModel:
    def __init__(self, boxes=BOXES):
        self.boxes = boxes

    def predict(self, source, imgsz, conf, device, verbose):
        data = torch.tensor(self.boxes, dtype=torch.float32)
        return [SimpleNamespace(boxes=SimpleNamespace(data=data)) for _ in source]


def blank_image():
    return np.zeros((IMG_H, IMG_W, 3), np.uint8)


def fake_detector(n_boxes, windows):
    det = np.tile(np.array([[10, 10, 50, 50, 0.9, 0]], np.float32), (n_boxes, 1))
    return lambda img: (det, np.array(windows, np.int32).reshape(-1, 4))


def fake_detectors():
    return {
        "full": fake_detector(2, []),
        "slice": fake_detector(3, [[0, 0, 640, 640]] * 6),
        "sahi": fake_detector(4, [[0, 0, 640, 640]] * 6),
    }


def make_trace():
    return adaptive_trace(FakeModel(), blank_image(), PARAMS, "cpu")


def test_figure_paths_use_new_names_per_image(tmp_path):
    paths = figure_paths(Path("data/0000117_a.jpg"), tmp_path)
    assert {k: v.name for k, v in paths.items()} == {
        "etape": "etape_0000117_a.png",
        "comparatie": "comparatie_metode_0000117_a.png",
        "unire": "unire_ferestre_0000117_a.png",
    }
    other = figure_paths(Path("data/other.jpg"), tmp_path)
    assert not {p.name for p in paths.values()} & {p.name for p in other.values()}
    assert not {p.name for p in paths.values()} & OLD_NAMES


def test_grid_windows_match_slicing_windows():
    grid = grid_windows(1360, 765, 640, 0.2)
    assert grid.dtype == np.int32
    assert grid.tolist() == [list(w) for w in slicing.windows(1360, 765, 640, 0.2)]
    assert len(grid) == 6


def test_sahi_windows_give_int32_boxes_inside_image():
    wins = sahi_windows(1360, 765, 640, 0.2)
    assert wins.dtype == np.int32
    assert len(wins) == 6
    assert (wins[:, [0, 1]] >= 0).all() and (wins[:, 2] <= 1360).all() and (wins[:, 3] <= 765).all()


def test_pixels_processed_counts_first_pass_plus_windows():
    assert pixels_processed(np.array([[0, 0, 1000, 1200]], np.int32), IMG_W, IMG_H) == pytest.approx(1.5)
    assert pixels_processed(np.zeros((0, 4), np.int32), IMG_W, IMG_H) == pytest.approx(1.0)


@pytest.mark.parametrize("with_clahe", [False, True], ids=["no_clahe", "with_clahe"])
def test_build_stages_figure_draws_every_window_and_detection(with_clahe):
    trace = make_trace()
    grid = grid_windows(IMG_W, IMG_H, 640, 0.2)
    fig = build_stages_figure(blank_image(), grid, trace, blank_image() if with_clahe else None)
    axes = fig.axes
    assert len(axes) == (5 if with_clahe else 4)
    assert [len(ax.patches) for ax in axes] == [0] * (len(axes) - 3) + [len(grid), len(trace.stages.windows), len(trace.detections)]
    plt.close(fig)


def test_build_methods_figure_titles_and_boxes():
    trace = make_trace()
    results = run_methods(blank_image(), {**fake_detectors(), "adaptive": adaptive_detector(trace)})
    fig = build_methods_figure(blank_image(), results)
    assert len(fig.axes) == 4
    for ax, r in zip(fig.axes, results):
        title = ax.get_title()
        assert METHOD_LABELS[r.mode] in title
        assert f"{len(r.det)} detectii" in title
        assert f"{len(r.windows)} ferestre" in title
        assert f"{pixels_processed(r.windows, IMG_W, IMG_H):.2f}x pixeli" in title
        assert not re.search(r"\bms\b", title)
        assert len(ax.patches) == len(r.det) + len(r.windows)
    plt.close(fig)


def test_methods_titles_show_pixel_factor_for_every_method():
    results = run_methods(blank_image(), {**fake_detectors(), "adaptive": adaptive_detector(make_trace())})
    titles = {r.mode: title_of(r) for r in results}
    assert titles["full"].endswith("1.00x pixeli")
    assert titles["slice"] == "Grila: 3 detectii, 6 ferestre, 2.02x pixeli"


def title_of(result):
    return method_title(result, IMG_W, IMG_H)


def merge_traces_for_test():
    two_boxes = [[400, 200, 720, 520, 0.9, 3], [800, 200, 1120, 520, 0.9, 3]]
    model = FakeModel(two_boxes)
    off = replace(PARAMS, merge_overlap=0.0)
    on = replace(PARAMS, merge_overlap=0.3, max_window=1280)
    return adaptive_trace(model, blank_image(), off, "cpu"), adaptive_trace(model, blank_image(), on, "cpu")


def test_build_merge_figure_has_windows_row_and_detections_row():
    before, after = merge_traces_for_test()
    assert len(before.stages.windows) == 2 and len(after.stages.windows) == 1
    fig = build_merge_figure(blank_image(), before, after)
    top_left, top_right, bottom_left, bottom_right = fig.axes
    assert len(fig.axes) == 4
    assert top_left.get_title() == f"Fara unire: 2 ferestre, {pixels_processed(before.stages.windows, IMG_W, IMG_H):.2f}x pixeli"
    assert top_right.get_title() == f"Cu unire: 1 ferestre, {pixels_processed(after.stages.windows, IMG_W, IMG_H):.2f}x pixeli"
    assert bottom_left.get_title() == f"Detectii: {len(before.detections)}"
    assert bottom_right.get_title() == f"Detectii: {len(after.detections)}"
    # sus: contur + umplere pentru fiecare fereastra; jos: cate o caseta pe detectie
    assert (len(top_left.patches), len(top_right.patches)) == (4, 2)
    assert (len(bottom_left.patches), len(bottom_right.patches)) == (len(before.detections), len(after.detections))
    plt.close(fig)


def test_merge_figure_window_fill_is_translucent_and_outline_is_kept():
    before, after = merge_traces_for_test()
    fig = build_merge_figure(blank_image(), before, after)
    outline, fill = fig.axes[0].patches[0], fig.axes[0].patches[-1]
    assert not outline.get_fill()
    assert fill.get_fill() and fill.get_alpha() == WINDOW_FILL_ALPHA
    plt.close(fig)


@pytest.mark.parametrize("with_clahe", [False, True], ids=["no_clahe", "with_clahe"])
def test_render_image_writes_three_pngs_without_touching_old_names(tmp_path, with_clahe):
    image = tmp_path / "foto.png"
    cv2.imwrite(str(image), blank_image())
    out = tmp_path / "figuri"
    args = SimpleNamespace(out=out, dpi=30, slice=640, overlap=0.2, device="cpu")
    merge_params = replace(PARAMS, merge_overlap=0.3, max_window=1280)
    setup = Setup(PARAMS, merge_params, FakeModel(), fake_detectors(), FakeModel() if with_clahe else None)
    render_image(image, args, setup)
    written = sorted(p.name for p in out.iterdir())
    assert written == ["comparatie_metode_foto.png", "etape_foto.png", "unire_ferestre_foto.png"]
    assert not set(written) & OLD_NAMES
    for p in out.iterdir():
        assert p.stat().st_size > 0
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_parse_args_defaults():
    args = parse_args(["--weights", "w.pt", "--image", "a.jpg", "b.jpg"])
    assert args.image == [Path("a.jpg"), Path("b.jpg")]
    assert args.weights_clahe is None
    assert args.conf == 0.25
    assert args.out == Path("results/figuri")
    assert args.adaptive_config == "configs/adaptive.yaml"
    assert args.merge_config == "configs/adaptive_merge.yaml"
    assert (args.slice, args.overlap, args.device) == (640, 0.2, None)
