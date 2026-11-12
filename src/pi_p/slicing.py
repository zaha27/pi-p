"""Grila de ferestre pătrate suprapuse, folosită la tăierea feliilor și la inferență."""

import numpy as np

Window = tuple[int, int, int, int]


def starts(length: int, size: int, step: int) -> list[int]:
    """Începuturile ferestrelor pe o latură; ultima e lipită de margine (1360, 640, 512 -> 0, 512, 720)."""
    if length <= size:
        return [0]
    return [*range(0, length - size, step), length - size]


def windows(w: int, h: int, size: int, overlap: float) -> list[Window]:
    """Ferestrele care acoperă o imagine `w` x `h`, cu vecinele suprapuse pe fracția `overlap`."""
    step = max(1, round(size * (1 - overlap)))
    wins = []
    for y in starts(h, size, step):
        for x in starts(w, size, step):
            wins.append((x, y, min(x + size, w), min(y + size, h)))
    return wins


def crop_boxes(xyxy: np.ndarray, win: Window, min_visibility: float) -> tuple[np.ndarray, np.ndarray]:
    """Chenarele vizibile cel puțin `min_visibility` în fereastră, în coordonatele ei."""
    x1, y1, x2, y2 = win

    clipped = xyxy.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(x1, x2)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(y1, y2)

    full_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    visible_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    keep = (visible_area > 0) & (visible_area >= min_visibility * full_area)

    origin = np.array([x1, y1, x1, y1], dtype=clipped.dtype)
    return keep, clipped[keep] - origin
