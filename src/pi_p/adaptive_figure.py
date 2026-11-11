import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from pi_p.adaptive import MASK_ON, AdaptiveParams, AdaptiveTrace, adaptive_trace, keep_confident, load_params

FIGSIZE = (18, 9)
COMPARISON_FIGSIZE = (16, 6)
DPI = 130
LINE_WIDTH = 1.5
WINDOW_LINE_WIDTH = 1.0
WINDOW_LINE_STYLE = "--"
MASK_CMAP = "gray"
COLOR_PRELIM = "#7E22CE"
COLOR_RECTS = "#1D4ED8"
COLOR_WINDOWS = "#B45309"
COLOR_DETECTIONS = "#15803D"


def read_image(path: Path) -> np.ndarray:
    # BGR, ca în predict.py: modelul așteaptă BGR, RGB e doar pentru afișare
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Imagine ilizibilă: {path}")
    return img


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(ax: Axes, boxes_xyxy: np.ndarray, color: str, linewidth: float = LINE_WIDTH, linestyle: str = "-") -> None:
    for x1, y1, x2, y2 in boxes_xyxy:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))


def box_panel(ax: Axes, img_rgb: np.ndarray, boxes_xyxy: np.ndarray, color: str, title: str) -> None:
    ax.imshow(img_rgb)
    draw_boxes(ax, boxes_xyxy, color)
    ax.set_title(title)
    ax.axis("off")


def mask_panel(ax: Axes, mask: np.ndarray, image_hw: tuple[int, int], title: str) -> None:
    h, w = image_hw
    # extent în pixeli ai imaginii mari: masca mică se întinde peste aceleași coordonate
    ax.imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=MASK_ON, extent=(0, w, h, 0))
    ax.set_title(title)
    ax.axis("off")


def build_figure(img_bgr: np.ndarray, trace: AdaptiveTrace) -> Figure:
    img = to_rgb(img_bgr)
    s = trace.stages
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    ax = axes.flat
    box_panel(ax[0], img, trace.prelim[:, :4].numpy(), COLOR_PRELIM, f"Detecție preliminară: {len(trace.prelim)} casete")
    mask_panel(ax[1], s.mask, img.shape[:2], f"Mască binară din {len(trace.prelim)} casete")
    mask_panel(ax[2], s.dilated, img.shape[:2], f"Mască dilatată: {len(s.rects)} pete")
    # rects sunt în pixeli ai măștii; scale îi duce în pixeli ai imaginii
    box_panel(ax[3], img, s.rects * s.scale, COLOR_RECTS, f"Dreptunghiuri: {len(s.rects)}")
    box_panel(ax[4], img, s.windows, COLOR_WINDOWS, f"Ferestre: {len(s.windows)}")
    box_panel(ax[5], img, trace.detections[:, :4].numpy(), COLOR_DETECTIONS, f"Detecții finale: {len(trace.detections)}")
    fig.tight_layout()
    return fig


def build_comparison_figure(img_bgr: np.ndarray, trace: AdaptiveTrace, params: AdaptiveParams) -> Figure:
    img = to_rgb(img_bgr)
    # fără ferestre = detecția pe imaginea întreagă, deja în trace.prelim
    plain = keep_confident(trace.prelim, params.conf)[:, :4].numpy()
    final = trace.detections[:, :4].numpy()
    fig, (left, right) = plt.subplots(1, 2, figsize=COMPARISON_FIGSIZE)
    box_panel(left, img, plain, COLOR_DETECTIONS, f"Fără ferestre (imgsz {params.prelim_imgsz}): {len(plain)} detecții")
    box_panel(right, img, final, COLOR_DETECTIONS, f"Metoda adaptivă: {len(final)} detecții, {len(trace.stages.windows)} ferestre")
    draw_boxes(right, trace.stages.windows, COLOR_WINDOWS, WINDOW_LINE_WIDTH, WINDOW_LINE_STYLE)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="fișier PNG")
    ap.add_argument("--comparison-out", type=Path, default=None, help="fișier PNG")
    ap.add_argument("--config", default="configs/adaptive.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()
    params = load_params(args.config, tuple(args.overrides))
    img = read_image(args.image)
    trace = adaptive_trace(YOLO(args.weights), img, params, args.device)
    save_figure(build_figure(img, trace), args.out)
    print(f"Figură în {args.out}")
    if args.comparison_out:
        save_figure(build_comparison_figure(img, trace, params), args.comparison_out)
        print(f"Comparație în {args.comparison_out}")


if __name__ == "__main__":
    main()
