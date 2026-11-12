import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pi_p.data import IMG_EXT
from pi_p.slicing import Window, windows


def ios(boxes: torch.Tensor) -> torch.Tensor:
    # IOS = aria comuna / aria chenarului mai mic; un chenar mic aflat intr-unul mare are IOS 1
    top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    common = (bottom_right - top_left).clamp(min=0).prod(2)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    smaller = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-6)
    return common / smaller


def merge(det: torch.Tensor, thr: float) -> torch.Tensor:
    det = det[det[:, 4].argsort(descending=True)]
    same_class = det[:, None, 5] == det[None, :, 5]
    duplicate = (ios(det[:, :4]) > thr) & same_class

    # de la cel mai sigur la cel mai nesigur: fiecare chenar pastrat ii elimina dublurile de dupa el
    keep = torch.ones(len(det), dtype=torch.bool)
    for i in range(len(det)):
        if keep[i]:
            keep[i + 1 :] &= ~duplicate[i, i + 1 :]
    return det[keep]


def drop_cut(det: torch.Tensor, win: Window, w: int, h: int, margin: int = 2) -> torch.Tensor:
    # taie obiecte doar marginile ferestrei din interiorul imaginii, nu si marginea imaginii
    x1, y1, x2, y2 = win
    win_w, win_h = x2 - x1, y2 - y1
    cut = torch.zeros(len(det), dtype=torch.bool)
    if x1 > 0:
        cut |= det[:, 0] <= margin
    if y1 > 0:
        cut |= det[:, 1] <= margin
    if x2 < w:
        cut |= det[:, 2] >= win_w - margin
    if y2 < h:
        cut |= det[:, 3] >= win_h - margin
    return det[~cut]


def to_image(det: torch.Tensor, win: Window, w: int, h: int) -> torch.Tensor:
    x1, y1 = win[0], win[1]
    offset = torch.tensor([x1, y1, x1, y1, 0, 0])
    return drop_cut(det, win, w, h) + offset


def sliced_predict(model: YOLO, img: np.ndarray, size: int, overlap: float, conf: float, iou: float, device) -> torch.Tensor:
    h, w = img.shape[:2]
    wins = windows(w, h, size, overlap)
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]

    # imaginea intreaga prinde obiectele mari, ferestrele pe cele mici
    preds = model.predict([img, *crops], imgsz=size, conf=conf, device=device, verbose=False)
    whole, *in_windows = [p.boxes.data.cpu() for p in preds]
    from_windows = [to_image(det, win, w, h) for det, win in zip(in_windows, wins)]
    return merge(torch.cat([whole, *from_windows]), iou)


def image_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
    if source.is_file() and source.suffix.lower() in IMG_EXT:
        return [source]
    raise SystemExit(f"--slice merge doar pe imagini sau foldere cu imagini: {source}")


def report(r: Results) -> None:
    counts = Counter(r.names[int(c)] for c in r.boxes.cls)
    print(f"{r.path}: {dict(counts)}")


def predict_whole(model: YOLO, args) -> None:
    results = model.predict(args.source, conf=args.conf, device=args.device, save=True, stream=True,
                            project=args.out.parent, name=args.out.name)
    for r in results:
        report(r)


def predict_sliced(model: YOLO, args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for path in image_paths(Path(args.source)):
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path}: imagine ilizibila, sarita")
            continue
        boxes = sliced_predict(model, img, args.slice, args.overlap, args.conf, args.iou, args.device)
        r = Results(img, path=str(path), names=model.names, boxes=boxes)
        cv2.imwrite(str(args.out / path.name), r.plot(line_width=1, labels=False))
        report(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("source", help="imagine, folder, video sau URL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la combinarea detectiilor din ferestre")
    ap.add_argument("--device", default=None)
    ap.add_argument("--slice", type=int, default=0, help="marimea ferestrei (0 = fara ferestre)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=Path("runs/predict"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    if args.slice:
        predict_sliced(model, args)
    else:
        predict_whole(model, args)


if __name__ == "__main__":
    main()
