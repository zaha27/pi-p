import csv
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pi_p.analytics import Analytics
from pi_p.app.render import congestion_overlay, draw_frame, draw_grid, heat_overlay, resize_to
from pi_p.data import COLORS
from pi_p.predict import sliced_predict
from pi_p.scoremap import ScoreMap
from pi_p.slicing import windows

DISPLAY_WIDTH = 1280
CONGESTION_WIDTH = 720
CONGESTION_EVERY = 1.0
PREDICT_CONF = 0.05
OVERLAP = 0.2
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 82]
GPU = threading.Lock()


def device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def model_choices() -> list[str]:
    return [str(p) for p in sorted(Path("runs").glob("*/weights/best.pt"))] + ["yolo26n.pt"]


def encode(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, JPEG)[1].tobytes()


@dataclass
class Params:
    conf: float = 0.25
    view: str = "detections"


class Session:
    def __init__(self, video: Path, weights: str, track: bool, slice_size: int, out_dir: Path):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise ValueError("Nu pot deschide video-ul")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dev = device()
        self.model = YOLO(weights)
        with GPU:
            self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=self.dev, verbose=False)
        self.scores = ScoreMap(self.model.predictor.model.model)
        self.slice = slice_size
        self.track = track and not slice_size
        self.params = Params()
        self.analytics = Analytics(self.model.names, self.width, self.height, self.fps)
        self.trails = defaultdict(lambda: deque(maxlen=30))
        self.out_dir = out_dir
        self.writer = None
        self.rows: list[dict] = []
        self.last_congestion = 0.0

    def meta(self, weights: str) -> dict:
        return {
            "type": "meta",
            "model": Path(weights).stem if Path(weights).name != "best.pt" else Path(weights).parents[1].name,
            "names": {int(k): v for k, v in self.model.names.items()},
            "colors": {int(k): "#%02x%02x%02x" % COLORS[int(k) % len(COLORS)] for k in self.model.names},
            "fps": self.fps,
            "total": self.total,
            "width": self.width,
            "height": self.height,
            "device": self.dev,
            "track": self.track,
        }

    def detect(self, frame: np.ndarray):
        if self.slice:
            det = sliced_predict(self.model, frame, self.slice, OVERLAP, PREDICT_CONF, 0.5, self.dev).numpy()
            return det[:, :4], det[:, 4], det[:, 5].astype(int), None
        if self.track:
            r = self.model.track(frame, conf=PREDICT_CONF, device=self.dev, persist=True, verbose=False)[0]
        else:
            r = self.model.predict(frame, conf=PREDICT_CONF, device=self.dev, verbose=False)[0]
        b = r.boxes.cpu()
        ids = b.id.int().numpy() if b.id is not None else None
        return b.xyxy.numpy(), b.conf.numpy(), b.cls.int().numpy(), ids

    def step(self) -> tuple[bytes, bytes | None, dict] | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        small, s = resize_to(frame, DISPLAY_WIDTH)
        with GPU:
            t = time.perf_counter()
            xyxy, conf, cls, ids = self.detect(frame)
            ms = (time.perf_counter() - t) * 1000
            cam = self.scores.heatmap(*small.shape[:2]) if self.params.view == "scores" else None
        keep = conf >= self.params.conf
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
        ids = ids[keep] if ids is not None else None
        self.analytics.update(xyxy, cls, ids, ms)

        base = small if cam is None else heat_overlay(small, cam)
        view = draw_frame(base, xyxy * s, conf, cls, ids, self.trails, self.model.names)
        if self.slice:
            draw_grid(view, [tuple(round(v * s) for v in win) for win in windows(self.width, self.height, self.slice, OVERLAP)])
        if self.writer is None:
            self.writer = self._open_writer(view.shape[1], view.shape[0])
        self.writer.write(view)
        counts = Counter(cls.tolist())
        self.rows.append({"frame": len(self.rows), **{name: counts.get(k, 0) for k, name in self.model.names.items()}})

        congestion = None
        now = time.perf_counter()
        if now - self.last_congestion >= CONGESTION_EVERY:
            self.last_congestion = now
            occ = self.analytics.occupancy()
            congestion = encode(congestion_overlay(resize_to(frame, CONGESTION_WIDTH)[0], occ, self.analytics.peak(occ)))
        return encode(view), congestion, self.analytics.summary()

    def _open_writer(self, w: int, h: int) -> cv2.VideoWriter:
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(self.out_dir / "detectii.mp4"), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if writer.isOpened():
                return writer
        raise RuntimeError("Nu pot crea fișierul video de ieșire")

    def close(self) -> list[str]:
        self.cap.release()
        if self.writer is None:
            return []
        self.writer.release()
        with open(self.out_dir / "numaratoare.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        return ["detectii.mp4", "numaratoare.csv"]
