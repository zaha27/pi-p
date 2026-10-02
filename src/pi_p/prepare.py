import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS_URL, TQDM
from ultralytics.utils.downloads import download
from ultralytics.utils.ops import xyxy2xywhn

from pi_p.data import ROOT, image_size, write_data_yaml, write_labels

RAW = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
FILL = 114


def parse(text: str) -> np.ndarray:
    rows = [[int(v) for v in line.strip().split(",")[:8]] for line in text.splitlines() if line.strip()]
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def convert(ann: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyxy = np.stack([ann[:, 0], ann[:, 1], ann[:, 0] + ann[:, 2], ann[:, 1] + ann[:, 3]], 1)
    xyxy = np.clip(xyxy, 0, [w, h, w, h])
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    cat = ann[:, 5]
    keep = valid & (ann[:, 4] != 0) & (cat >= 1) & (cat <= 10)
    return cat[keep] - 1, xyxy[keep], xyxy[valid & ~keep]


def mask_ignored(img: np.ndarray, keep: np.ndarray, ignore: np.ndarray) -> bool:
    m = np.zeros(img.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in ignore:
        m[y1:y2, x1:x2] = True
    for x1, y1, x2, y2 in keep:
        m[y1:y2, x1:x2] = False
    img[m] = FILL
    return bool(m.any())


def process(raw: Path, root: Path, split: str) -> dict:
    img_dir = root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "objects": 0, "ignored_regions": 0, "masked_images": 0}
    for ann_file in TQDM(sorted((raw / "annotations").glob("*.txt")), desc=split):
        dst = img_dir / f"{ann_file.stem}.jpg"
        src = raw / "images" / dst.name
        if not src.exists():
            src = dst
        w, h = image_size(src)
        cls, keep, ignore = convert(parse(ann_file.read_text()), w, h)
        img = cv2.imread(str(src)) if len(ignore) else None
        if img is not None and mask_ignored(img, keep, ignore):
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["masked_images"] += 1
            if src != dst:
                src.unlink()
        elif src != dst:
            shutil.move(src, dst)
        labels = np.column_stack([cls, xyxy2xywhn(keep.astype(np.float64), w=w, h=h)])
        write_labels(root / "labels" / split / f"{ann_file.stem}.txt", labels)
        stats["images"] += 1
        stats["objects"] += len(cls)
        stats["ignored_regions"] += len(ignore)
    ann_dir = root / "annotations" / split
    ann_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(raw / "annotations", ann_dir)
    shutil.rmtree(raw)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--splits", nargs="+", default=list(RAW), choices=list(RAW))
    args = ap.parse_args()
    root = args.root
    todo = [s for s in args.splits if (root / RAW[s]).is_dir() or not (root / "labels" / s).is_dir()]
    missing = [s for s in todo if not (root / RAW[s]).is_dir()]
    if missing:
        download([f"{ASSETS_URL}/{RAW[s]}.zip" for s in missing], dir=root, delete=True, threads=len(missing))
    for s in todo:
        print(s, process(root / RAW[s], root, s))
    print(f"Dataset gata: {write_data_yaml(root, args.splits)}")


if __name__ == "__main__":
    main()
