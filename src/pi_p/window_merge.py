import numpy as np


def overlap_ratio(a: list[int], b: list[int]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    # raportat la fereastra mai mica, ca o fereastra aproape continuta in alta sa se uneasca
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller


def union_box(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def merge_score(a: list[int], b: list[int], overlap: float, max_window: int) -> float:
    ratio = overlap_ratio(a, b)
    x1, y1, x2, y2 = union_box(a, b)
    return ratio if ratio > overlap and x2 - x1 <= max_window and y2 - y1 <= max_window else 0.0


def best_pair(wins: list[list[int]], overlap: float, max_window: int) -> tuple[int, int] | None:
    best, pair = 0.0, None
    for i in range(len(wins)):
        for j in range(i + 1, len(wins)):
            score = merge_score(wins[i], wins[j], overlap, max_window)
            if score > best:
                best, pair = score, (i, j)
    return pair


def merge_windows(windows: np.ndarray, overlap: float, max_window: int) -> np.ndarray:
    # overlap <= 0 inseamna unirea dezactivata
    if overlap <= 0:
        return windows
    wins = windows.tolist()
    while (pair := best_pair(wins, overlap, max_window)) is not None:
        i, j = pair
        wins[i] = union_box(wins[i], wins[j])
        del wins[j]
    return np.array(wins, dtype=np.int32).reshape(-1, 4)
