import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pi_p.data import IMG_EXT
from pi_p.slicing import windows


def merge(det: torch.Tensor, thr: float) -> torch.Tensor:
    det = det[det[:, 4].argsort(descending=True)]
    lt = torch.maximum(det[:, None, :2], det[None, :, :2])
    rb = torch.minimum(det[:, None, 2:4], det[None, :, 2:4])
    inter = (rb - lt).clamp(min=0).prod(2)
    area = (det[:, 2:4] - det[:, :2]).prod(1)
    ios = inter / torch.minimum(area[:, None], area[None, :]).clamp(min=1e-6)
    suppress = (ios > thr) & (det[:, None, 5] == det[None, :, 5])
    keep = torch.ones(len(det), dtype=torch.bool)
    for i in range(len(det)):
        if keep[i]:
            keep[i + 1 :] &= ~suppress[i, i + 1 :]
    return det[keep]


def drop_cut(det: torch.Tensor, win: tuple[int, int, int, int], w: int, h: int, margin: int = 2) -> torch.Tensor:
    x1, y1, x2, y2 = win
    cut = torch.zeros(len(det), dtype=torch.bool)
    if x1 > 0:
        cut |= det[:, 0] <= margin
    if y1 > 0:
        cut |= det[:, 1] <= margin
    if x2 < w:
        cut |= det[:, 2] >= x2 - x1 - margin
    if y2 < h:
        cut |= det[:, 3] >= y2 - y1 - margin
    return det[~cut]


def sliced_predict(model: YOLO, img: np.ndarray, size: int, overlap: float, conf: float, iou: float, device) -> torch.Tensor:
    h, w = img.shape[:2]
    wins = windows(w, h, size, overlap)
    preds = model.predict([img, *(img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins)], imgsz=size, conf=conf, device=device, verbose=False)
    dets = [preds[0].boxes.data.cpu()]
    for p, win in zip(preds[1:], wins):
        dets.append(drop_cut(p.boxes.data.cpu(), win, w, h) + torch.tensor([win[0], win[1], win[0], win[1], 0, 0]))
    return merge(torch.cat(dets), iou)


def image_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
    if source.is_file() and source.suffix.lower() in IMG_EXT:
        return [source]
    raise SystemExit(f"--slice merge doar pe imagini sau foldere cu imagini: {source}")


def report(r: Results) -> None:
    print(f"{r.path}: {dict(Counter(r.names[int(c)] for c in r.boxes.cls))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("source", help="imagine, folder, video sau URL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la combinarea detecțiilor din ferestre")
    ap.add_argument("--device", default=None)
    ap.add_argument("--slice", type=int, default=0, help="mărimea ferestrei pentru inferență pe bucăți (0 = dezactivat)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=Path("runs/predict"))
    args = ap.parse_args()
    model = YOLO(args.weights)
    if not args.slice:
        for r in model.predict(args.source, conf=args.conf, device=args.device, save=True, stream=True, project=args.out.parent, name=args.out.name):
            report(r)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for p in image_paths(Path(args.source)):
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p}: imagine ilizibilă, sărită")
            continue
        r = Results(img, path=str(p), names=model.names, boxes=sliced_predict(model, img, args.slice, args.overlap, args.conf, args.iou, args.device))
        cv2.imwrite(str(args.out / p.name), r.plot(line_width=1, labels=False))
        report(r)


if __name__ == "__main__":
    main()
