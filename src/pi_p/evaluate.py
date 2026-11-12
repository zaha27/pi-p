import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pi_p import coco
from pi_p.data import images, label_path, read_labels
from pi_p.metrics import mean_ap
from pi_p.predict import sliced_predict
from pi_p.stats import read_groups


def ground_truth(img_path: Path, w: int, h: int) -> np.ndarray:
    lab = read_labels(label_path(img_path))
    cls, cx, cy, bw, bh = lab.T
    return np.stack([cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)


def sahi_detect(sahi_model, img: np.ndarray, args) -> np.ndarray:
    from sahi.predict import get_sliced_prediction

    res = get_sliced_prediction(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sahi_model,
        slice_height=args.slice, slice_width=args.slice,
        overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
        perform_standard_pred=True, postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS", postprocess_match_threshold=args.iou, verbose=0,
    )
    rows = [[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in res.object_prediction_list]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def detect(model, img: np.ndarray, args) -> np.ndarray:
    if args.mode == "full":
        return model.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0].boxes.data.cpu().numpy()
    if args.mode == "slice":
        return sliced_predict(model, img, args.slice, args.overlap, args.conf, args.iou, args.device).numpy()
    return sahi_detect(model, img, args)


def load_model(args):
    if args.mode != "sahi":
        return YOLO(args.weights)
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, confidence_threshold=args.conf,
        image_size=args.slice, device=args.device or "cpu",
    )


def evaluate(args) -> None:
    model = load_model(args)
    paths = images(Path(args.data).parent, args.split)[: args.limit or None]
    results, done, seconds = [], [], 0.0
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p}: imagine ilizibilă, sărită")
            continue
        start = time.perf_counter()
        det = detect(model, img, args)
        seconds += time.perf_counter() - start
        results.append((det, ground_truth(p, img.shape[1], img.shape[0])))
        done.append(p.name)
        if i % 50 == 0:
            print(f"{i}/{len(paths)}")
    map50, map50_95, per_class = mean_ap(results)
    print(f"mod {args.mode}, {len(results)} imagini, {1000 * seconds / max(len(results), 1):.0f} ms/imagine")
    print(f"mAP50-95: {map50_95:.3f}  mAP50: {map50:.3f}")
    names = YOLO(args.weights).names if args.mode == "sahi" else model.names
    for c, ap in per_class.items():
        print(f"  {names[c]:>16}: {ap:.3f}")
    if args.groups:
        groups = read_groups(args.groups)
        for g in sorted(set(groups.values())):
            subset = [r for r, name in zip(results, done) if groups.get(name) == g]
            if subset:
                g50, g50_95, _ = mean_ap(subset)
                print(f"grupa {g} ({len(subset)} imagini): mAP50-95: {g50_95:.3f}  mAP50: {g50:.3f}")


def validate(args) -> None:
    model = YOLO(args.weights)
    data, classes = args.data, None
    if args.coco:
        if len(model.names) != 80:
            raise SystemExit(f"--coco cere un model cu cele 80 de clase COCO, {args.weights} are {len(model.names)}")
        data = coco.build(Path(args.data).parent, "val", model.names)
        classes = sorted(set(coco.TO_COCO.values()))
    m = model.val(data=str(data), imgsz=args.imgsz, device=args.device, classes=classes)
    print(f"mAP50-95: {m.box.map:.3f}  mAP50: {m.box.map50:.3f}")
    for c in m.box.ap_class_index:
        print(f"  {m.names[c]:>16}: {m.box.maps[c]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="ex. yolo26n.pt (baseline COCO, cu --coco) sau best.pt")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--coco", action="store_true", help="model cu clase COCO: mapează etichetele VisDrone pe COCO")
    ap.add_argument("--mode", choices=["val", "full", "slice", "sahi"], default="val",
                    help="val = model.val() Ultralytics; full/slice/sahi = mAP calculat de noi pe imaginea întreagă")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.001, help="același prag mic pentru toate variantele")
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la combinarea detecțiilor din ferestre")
    ap.add_argument("--slice", type=int, default=640, help="mărimea ferestrei pentru slice și sahi")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--groups", type=Path, default=None, help="stats.csv de la pi-stats: mAP și pe fiecare grupă")
    ap.add_argument("--limit", type=int, default=0, help="doar primele N imagini (0 = toate)")
    args = ap.parse_args()
    if args.mode == "val":
        validate(args)
    elif args.coco:
        raise SystemExit("--coco merge doar cu --mode val")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
