from collections import deque

import cv2
import numpy as np

from pi_p.analytics import JAM_LEVEL
from pi_p.data import COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONGESTION_STOPS = [(0.0, (242, 209, 107)), (0.45, (232, 131, 58)), (0.8, (179, 38, 30)), (1.0, (92, 15, 15))]


def color(cls: int) -> tuple[int, int, int]:
    r, g, b = COLORS[cls % len(COLORS)]
    return b, g, r


def resize_to(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= width:
        return img, 1.0
    s = width / w
    return cv2.resize(img, (width, round(h * s)), interpolation=cv2.INTER_AREA), s


def draw_frame(img: np.ndarray, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, ids: np.ndarray | None,
               trails: dict[int, deque], names: dict[int, str]) -> np.ndarray:
    out = img.copy()
    for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(xyxy.astype(int), conf, cls)):
        col = color(int(k))
        if ids is not None:
            trail = trails[int(ids[i])]
            trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
            cv2.polylines(out, [np.array(trail, dtype=np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1 + round(float(c) * 2), cv2.LINE_AA)
        tag = names[int(k)] + (f" {ids[i]}" if ids is not None else "")
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.36, 1)
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y2
        cv2.rectangle(out, (x1, ty), (x1 + tw + 4, ty + th + 4), col, -1)
        cv2.putText(out, tag, (x1 + 2, ty + th + 1), FONT, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_grid(img: np.ndarray, wins: list[tuple[int, int, int, int]]) -> None:
    for x1, y1, x2, y2 in wins:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1, cv2.LINE_AA)


def heat_overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(img, 0.45, colored, 0.55, 0)


def congestion_lut() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    stops = np.array([s for s, _ in CONGESTION_STOPS])
    rgb = np.array([c for _, c in CONGESTION_STOPS], dtype=np.float32)
    lut = np.stack([np.interp(x, stops, rgb[:, i]) for i in range(3)], 1)
    return lut[:, ::-1].astype(np.uint8)


LUT = congestion_lut()


def congestion_overlay(img: np.ndarray, occupancy: np.ndarray, peak: tuple[float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base = np.repeat((255 - (255 - gray) * 0.4)[..., None], 3, 2)
    level = np.clip(cv2.resize(occupancy, (w, h), interpolation=cv2.INTER_CUBIC) / JAM_LEVEL, 0, 1)
    colored = LUT[(level * 255).astype(np.uint8)].astype(np.float32)
    alpha = np.clip((level - 0.05) * 1.6, 0, 0.92)[..., None]
    out = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    for lv in (0.25, 0.5, 0.75):
        contours, _ = cv2.findContours((level >= lv).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (40, 30, 110), 1, cv2.LINE_AA)
    px, py, value = peak
    if value / JAM_LEVEL > 0.2:
        cv2.putText(out, "PEAK", (int(px * w) - 16, int(py * h) - 8), FONT, 0.38, (40, 30, 110), 1, cv2.LINE_AA)
    return out
