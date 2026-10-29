from pathlib import Path

import numpy as np
import yaml
from PIL import Image

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
COLORS = [(142, 85, 114), (212, 119, 138), (122, 139, 58), (42, 157, 143), (233, 162, 59),
          (200, 85, 61), (176, 141, 87), (92, 107, 115), (61, 90, 152), (74, 165, 192)]
ROOT = Path("datasets/VisDrone")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def images(root: Path, split: str) -> list[Path]:
    d = root / "images" / split
    if not d.is_dir():
        raise SystemExit(f"Lipsește {d}, rulează întâi pi-prepare")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(img: Path) -> Path:
    return img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt"


def read_labels(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(path, ndmin=2, dtype=np.float32)


def write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def write_data_yaml(root: Path, splits: list[str]) -> Path:
    cfg = {"path": str(root.resolve()), **{s: f"images/{s}" for s in splits}, "names": dict(enumerate(NAMES))}
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
