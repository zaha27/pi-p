import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from pi_p import evaluate, slicing
from pi_p.adaptive import AdaptiveParams, AdaptiveTrace, adaptive_trace, load_params
from pi_p.adaptive_figure import (
    COLOR_DETECTIONS,
    COLOR_WINDOWS,
    WINDOW_LINE_STYLE,
    WINDOW_LINE_WIDTH,
    box_panel,
    draw_boxes,
    read_image,
    to_rgb,
)
from pi_p.enhance import clahe

PANEL_WIDTH = 6.0
STAGES_HEIGHT = 4.8
FIGSIZE_METHODS = (24, 13.5)
FIGSIZE_MERGE = (24, 14)
WINDOW_FILL_ALPHA = 0.06
DEFAULT_DPI = 150
TITLE_SIZE = 20
SLIDE_RC = {"axes.titlesize": TITLE_SIZE}
COLOR_GRID = "#0E7490"
NO_BOXES = np.zeros((0, 4), np.int32)
METHOD_ORDER = ["full", "slice", "sahi", "adaptive"]
METHOD_LABELS = {"full": "Imagine întreagă", "slice": "Grilă", "sahi": "SAHI", "adaptive": "Ferestre adaptive"}
WINDOW_COLORS = {"full": COLOR_GRID, "slice": COLOR_GRID, "sahi": COLOR_GRID, "adaptive": COLOR_WINDOWS}

Detector = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class MethodResult(NamedTuple):
    mode: str
    det: np.ndarray
    windows: np.ndarray


class Setup(NamedTuple):
    params: AdaptiveParams
    merge_params: AdaptiveParams
    model: YOLO
    detectors: dict[str, Detector]
    clahe_model: YOLO | None


def figure_paths(image: Path, out: Path) -> dict[str, Path]:
    return {
        "etape": out / f"etape_{image.stem}.png",
        "comparatie": out / f"comparatie_metode_{image.stem}.png",
        "unire": out / f"unire_ferestre_{image.stem}.png",
    }


def no_windows(w: int, h: int, size: int, overlap: float) -> np.ndarray:
    return NO_BOXES


def grid_windows(w: int, h: int, size: int, overlap: float) -> np.ndarray:
    return np.array(slicing.windows(w, h, size, overlap), dtype=np.int32).reshape(-1, 4)


def sahi_windows(w: int, h: int, size: int, overlap: float) -> np.ndarray:
    from sahi.slicing import get_slice_bboxes

    # SAHI își taie singur imaginea, cu o grilă ușor diferită de slicing.windows
    boxes = get_slice_bboxes(h, w, slice_height=size, slice_width=size, overlap_height_ratio=overlap, overlap_width_ratio=overlap)
    return np.array(boxes, dtype=np.int32).reshape(-1, 4)


WINDOWS_OF = {"full": no_windows, "slice": grid_windows, "sahi": sahi_windows}


def pixels_processed(windows: np.ndarray, w: int, h: int) -> float:
    return 1 + evaluate.window_area(windows, w, h)


def method_title(result: MethodResult, w: int, h: int) -> str:
    return (
        f"{METHOD_LABELS[result.mode]}: {len(result.det)} detecții, {len(result.windows)} ferestre, "
        f"{pixels_processed(result.windows, w, h):.2f}x pixeli"
    )


def fill_boxes(ax: Axes, boxes_xyxy: np.ndarray, color: str, alpha: float) -> None:
    for x1, y1, x2, y2 in boxes_xyxy:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=color, edgecolor="none", alpha=alpha))


def windows_panel(ax: Axes, img_rgb: np.ndarray, windows: np.ndarray, title: str) -> None:
    # umplere transparentă peste contur: suprapunerile ies mai închise la culoare
    box_panel(ax, img_rgb, windows, COLOR_WINDOWS, title)
    fill_boxes(ax, windows, COLOR_WINDOWS, WINDOW_FILL_ALPHA)


def method_panel(ax: Axes, img_rgb: np.ndarray, result: MethodResult, title: str) -> None:
    box_panel(ax, img_rgb, result.det[:, :4], COLOR_DETECTIONS, title)
    draw_boxes(ax, result.windows, WINDOW_COLORS[result.mode], WINDOW_LINE_WIDTH, WINDOW_LINE_STYLE)


def build_stages_figure(img_bgr: np.ndarray, grid: np.ndarray, trace: AdaptiveTrace, clahe_img: np.ndarray | None = None) -> Figure:
    panels = [(to_rgb(img_bgr), NO_BOXES, COLOR_DETECTIONS, "Imaginea originală")]
    if clahe_img is not None:
        panels.append((to_rgb(clahe_img), NO_BOXES, COLOR_DETECTIONS, "După CLAHE"))
    shown = panels[-1][0]
    panels += [
        (shown, grid, COLOR_GRID, f"Grila de ferestre: {len(grid)}"),
        (shown, trace.stages.windows, COLOR_WINDOWS, f"Ferestre adaptive: {len(trace.stages.windows)}"),
        (shown, trace.detections[:, :4].numpy(), COLOR_DETECTIONS, f"Detecții finale: {len(trace.detections)}"),
    ]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(PANEL_WIDTH * len(panels), STAGES_HEIGHT))
        for ax, panel in zip(axes.flat, panels):
            box_panel(ax, *panel)
        fig.tight_layout()
    return fig


def build_methods_figure(img_bgr: np.ndarray, results: list[MethodResult]) -> Figure:
    img = to_rgb(img_bgr)
    h, w = img.shape[:2]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_METHODS)
        for ax, result in zip(axes.flat, results):
            method_panel(ax, img, result, method_title(result, w, h))
        fig.tight_layout()
    return fig


def build_merge_figure(img_bgr: np.ndarray, before: AdaptiveTrace, after: AdaptiveTrace) -> Figure:
    img = to_rgb(img_bgr)
    h, w = img.shape[:2]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_MERGE)
        for ax, (name, trace) in zip(axes[0], (("Fără unire", before), ("Cu unire", after))):
            wins = trace.stages.windows
            windows_panel(ax, img, wins, f"{name}: {len(wins)} ferestre, {pixels_processed(wins, w, h):.2f}x pixeli")
        for ax, trace in zip(axes[1], (before, after)):
            box_panel(ax, img, trace.detections[:, :4].numpy(), COLOR_DETECTIONS, f"Detecții: {len(trace.detections)}")
        fig.tight_layout()
    return fig


def adaptive_detector(trace: AdaptiveTrace) -> Detector:
    def detect(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return trace.detections.numpy(), trace.stages.windows

    return detect


def run_methods(img: np.ndarray, detectors: dict[str, Detector]) -> list[MethodResult]:
    return [MethodResult(mode, *detectors[mode](img)) for mode in METHOD_ORDER]


def method_args(args, params: AdaptiveParams, mode: str) -> SimpleNamespace:
    # câmpurile pe care le citesc evaluate.load_model și evaluate.detect
    return SimpleNamespace(
        mode=mode, weights=args.weights, conf=args.conf, device=args.device, imgsz=params.prelim_imgsz,
        slice=args.slice, overlap=args.overlap, iou=params.iou, params=params,
    )


def detect_method(model, ns: SimpleNamespace, windows_fn: Callable, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    return evaluate.detect(model, img, ns)[0], windows_fn(w, h, ns.slice, ns.overlap)


def make_detectors(args, params: AdaptiveParams, yolo: YOLO) -> dict[str, Detector]:
    models = {"full": yolo, "slice": yolo, "sahi": evaluate.load_model(method_args(args, params, "sahi"))}
    return {mode: partial(detect_method, models[mode], method_args(args, params, mode), WINDOWS_OF[mode]) for mode in WINDOWS_OF}


def make_setup(args) -> Setup:
    params = load_params(args.adaptive_config, (f"conf={args.conf}",))
    merge_params = load_params(args.merge_config, (f"conf={args.conf}",))
    yolo = evaluate.load_model(method_args(args, params, "full"))
    clahe_model = YOLO(args.weights_clahe) if args.weights_clahe else None
    return Setup(params, merge_params, yolo, make_detectors(args, params, yolo), clahe_model)


def stages_figure(img: np.ndarray, args, setup: Setup, trace: AdaptiveTrace) -> Figure:
    h, w = img.shape[:2]
    grid = grid_windows(w, h, args.slice, args.overlap)
    if setup.clahe_model is None:
        return build_stages_figure(img, grid, trace)
    # figura de etape arată ce vede modelul CLAHE: imaginea după CLAHE
    enhanced = clahe(img)
    return build_stages_figure(img, grid, adaptive_trace(setup.clahe_model, enhanced, setup.params, args.device), enhanced)


def merge_traces(img: np.ndarray, args, setup: Setup) -> tuple[AdaptiveTrace, AdaptiveTrace]:
    # fără unire = aceiași parametri cu merge_overlap=0, ca să difere doar unirea; ambele rulează modelul
    off = replace(setup.merge_params, merge_overlap=0.0)
    return adaptive_trace(setup.model, img, off, args.device), adaptive_trace(setup.model, img, setup.merge_params, args.device)


def save_slide(fig: Figure, out: Path, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def render_image(path: Path, args, setup: Setup) -> None:
    img = read_image(path)
    trace = adaptive_trace(setup.model, img, setup.params, args.device)
    methods = run_methods(img, {**setup.detectors, "adaptive": adaptive_detector(trace)})
    figures = {
        "etape": stages_figure(img, args, setup, trace),
        "comparatie": build_methods_figure(img, methods),
        "unire": build_merge_figure(img, *merge_traces(img, args, setup)),
    }
    for name, out in figure_paths(path, args.out).items():
        save_slide(figures[name], out, args.dpi)
        print(f"{name}: {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="modelul principal")
    ap.add_argument("--weights-clahe", default=None, help="model antrenat pe imagini CLAHE; adaugă pasul CLAHE în figura de etape")
    ap.add_argument("--image", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/figuri"), help="director")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--adaptive-config", default="configs/adaptive.yaml")
    ap.add_argument("--merge-config", default="configs/adaptive_merge.yaml")
    ap.add_argument("--slice", type=int, default=640, help="mărimea ferestrei pentru grilă și SAHI")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    setup = make_setup(args)
    for path in args.image:
        render_image(path, args, setup)


if __name__ == "__main__":
    main()
