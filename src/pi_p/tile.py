import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import TQDM
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn

from pi_p.data import ROOT, images, label_path, read_labels, write_data_yaml, write_labels
from pi_p.slicing import crop_boxes, windows


def tile_image(img_path: Path, dst: Path, size: int, overlap: float, min_visibility: float, empty: float, full: bool) -> int:
    split = img_path.parent.name
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = read_labels(label_path(img_path))
    xyxy = xywhn2xyxy(lbl[:, 1:], w=w, h=h)
    rng = random.Random(img_path.stem)
    n = 0
    if full:
        shutil.copy2(img_path, dst / "images" / split / img_path.name)
        write_labels(dst / "labels" / split / f"{img_path.stem}.txt", lbl)
        n += 1
    for win in windows(w, h, size, overlap):
        keep, boxes = crop_boxes(xyxy, win, min_visibility)
        if not keep.any() and rng.random() >= empty:
            continue
        x1, y1, x2, y2 = win
        name = f"{img_path.stem}_{x1}_{y1}"
        cv2.imwrite(str(dst / "images" / split / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_labels(dst / "labels" / split / f"{name}.txt", np.column_stack([lbl[keep, 0], xyxy2xywhn(boxes, w=x2 - x1, h=y2 - y1)]))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT)
    ap.add_argument("--dst", type=Path, default=Path("datasets/VisDrone-tiled"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--empty", type=float, default=0.1, help="fracția de tile-uri fără obiecte păstrate")
    ap.add_argument("--no-full", action="store_true", help="nu include și imaginea întreagă")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not 0 <= args.overlap < 1 or not 0 < args.min_visibility <= 1:
        raise SystemExit("--overlap trebuie în [0, 1), --min-visibility în (0, 1]")
    if args.dst.exists():
        raise SystemExit(f"{args.dst} există deja; șterge-l sau alege alt --dst")
    fn = partial(tile_image, dst=args.dst, size=args.size, overlap=args.overlap,
                 min_visibility=args.min_visibility, empty=args.empty, full=not args.no_full)
    for split in args.splits:
        imgs = images(args.src, split)
        (args.dst / "images" / split).mkdir(parents=True)
        (args.dst / "labels" / split).mkdir(parents=True)
        with ThreadPoolExecutor(args.workers) as pool:
            n = sum(TQDM(pool.map(fn, imgs), total=len(imgs), desc=split))
        print(f"{split}: {len(imgs)} imagini -> {n} tile-uri")
    print(f"Dataset gata: {write_data_yaml(args.dst, args.splits)}")


if __name__ == "__main__":
    main()
