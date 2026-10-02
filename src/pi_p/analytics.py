from collections import Counter, deque

import cv2
import numpy as np

WINDOW_SECONDS = 10
HALF_LIFE_SECONDS = 20
GRID = 96
TRAIL = 8
SPARK_POINTS = 60
TIMELINE_POINTS = 120
JAM_LEVEL = 0.5


class Analytics:
    def __init__(self, names: dict[int, str], width: int, height: int, fps: float):
        self.names = names
        self.width, self.height, self.fps = width, height, fps
        self.history: deque[Counter] = deque(maxlen=max(2, round(WINDOW_SECONDS * fps)))
        self.classes: set[int] = set()
        self.tracks: dict[int, deque] = {}
        self.seen: set[int] = set()
        self.flow: list[tuple[float, float, float, float, int]] = []
        self.scale = GRID / max(width, height)
        self.dwell = np.zeros((max(1, round(height * self.scale)), max(1, round(width * self.scale))), np.float32)
        self.decay = 0.5 ** (1 / (HALF_LIFE_SECONDS * fps))
        self.frame = 0
        self.inference_ms = 0.0

    def update(self, xyxy: np.ndarray, cls: np.ndarray, ids: np.ndarray | None, inference_ms: float) -> None:
        self.frame += 1
        self.inference_ms = inference_ms
        self.history.append(Counter(cls.tolist()))
        self.classes.update(cls.tolist())
        self._accumulate(xyxy)
        self._update_tracks(xyxy, cls, ids)

    def _accumulate(self, xyxy: np.ndarray) -> None:
        self.dwell *= self.decay
        gh, gw = self.dwell.shape
        cells = np.floor(xyxy * self.scale).astype(int)
        for x1, y1, x2, y2 in cells:
            self.dwell[max(y1, 0) : min(y2 + 1, gh), max(x1, 0) : min(x2 + 1, gw)] += 1 - self.decay

    def _update_tracks(self, xyxy: np.ndarray, cls: np.ndarray, ids: np.ndarray | None) -> None:
        self.flow = []
        if ids is None:
            return
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        for (cx, cy), k, tid in zip(centers.tolist(), cls, ids):
            trail = self.tracks.setdefault(int(tid), deque(maxlen=TRAIL))
            trail.append((self.frame, cx, cy))
            self.seen.add(int(tid))
            if len(trail) > 1:
                f0, x0, y0 = trail[0]
                dt = self.frame - f0
                self.flow.append((cx, cy, (cx - x0) / dt, (cy - y0) / dt, int(k)))
        for tid in [t for t, tr in self.tracks.items() if self.frame - tr[-1][0] > TRAIL * 4]:
            del self.tracks[tid]

    def occupancy(self) -> np.ndarray:
        return cv2.GaussianBlur(self.dwell, (0, 0), 1.0)

    @staticmethod
    def peak(occ: np.ndarray) -> tuple[float, float, float]:
        py, px = np.unravel_index(int(occ.argmax()), occ.shape)
        return (px + 0.5) / occ.shape[1], (py + 0.5) / occ.shape[0], float(occ.max())

    def summary(self) -> dict:
        hist = list(self.history)
        current = hist[-1] if hist else Counter()
        in_frame = sum(current.values())
        order = sorted(self.classes)
        spark = hist[:: max(1, len(hist) // SPARK_POINTS)]
        timeline = hist[:: max(1, len(hist) // TIMELINE_POINTS)]
        occ = self.occupancy()
        speeds = [float(np.hypot(vx, vy)) for _, _, vx, vy, _ in self.flow]
        return {
            "frame": self.frame,
            "in_frame": in_frame,
            "unique": len(self.seen),
            "mean_flow": float(np.mean(speeds)) if speeds else 0.0,
            "inference_ms": self.inference_ms,
            "classes": [
                {"id": k, "count": current.get(k, 0), "share": current.get(k, 0) / in_frame if in_frame else 0.0,
                 "spark": [h.get(k, 0) for h in spark]}
                for k in order
            ],
            "timeline": {"seconds": len(hist) / self.fps, "series": [[h.get(k, 0) for h in timeline] for k in order]},
            "flow": [[x / self.width, y / self.height, vx / self.width, vy / self.height, k] for x, y, vx, vy, k in self.flow],
            "saturation": float((occ >= JAM_LEVEL).mean()),
            "peak": list(self.peak(occ)),
        }
