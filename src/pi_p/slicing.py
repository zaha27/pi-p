import numpy as np


def starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    return [*range(0, length - size, step), length - size]


def windows(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = max(1, round(size * (1 - overlap)))
    return [(x, y, min(x + size, w), min(y + size, h)) for y in starts(h, size, step) for x in starts(w, size, step)]


def crop_boxes(xyxy: np.ndarray, win: tuple[int, int, int, int], min_visibility: float) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = win
    c = xyxy.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x1, x2)
    c[:, [1, 3]] = c[:, [1, 3]].clip(y1, y2)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    vis = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    keep = (vis > 0) & (vis >= min_visibility * area)
    return keep, c[keep] - np.array([x1, y1, x1, y1], dtype=c.dtype)
