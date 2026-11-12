import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from pi_p import coco
from pi_p.adaptive import adaptive_predict, load_params
from pi_p.data import images, label_path, read_labels
from pi_p.metrics import mean_ap
from pi_p.predict import sliced_predict
from pi_p.stats import read_groups

WARMUP = 3
MAX_DET = 300
NO_WINDOWS = np.zeros((0, 4), np.int32)


def ground_truth(img_path: Path, w: int, h: int) -> np.ndarray:
    lab = read_labels(label_path(img_path))
    return np.column_stack([lab[:, 0], xywhn2xyxy(lab[:, 1:], w=w, h=h)])


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


def detect(model, img: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    if args.mode == "full":
        return model.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0].boxes.data.cpu().numpy(), NO_WINDOWS
    if args.mode == "slice":
        return sliced_predict(model, img, args.slice, args.overlap, args.conf, args.iou, args.device).numpy(), NO_WINDOWS
    if args.mode == "adaptive":
        det, windows = adaptive_predict(model, img, args.params, args.device)
        return det.numpy(), windows
    return sahi_detect(model, img, args), NO_WINDOWS


def load_model(args):
    if args.mode != "sahi":
        return YOLO(args.weights)
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, confidence_threshold=args.conf,
        image_size=args.slice, device=args.device or "cpu",
    )


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def window_area(windows: np.ndarray, w: int, h: int) -> float:
    win = windows.astype(np.int64)
    return float(((win[:, 2] - win[:, 0]) * (win[:, 3] - win[:, 1])).sum()) / (w * h)


def evaluate(args) -> None:
    model = load_model(args)
    paths = images(Path(args.data).parent, args.split)[: args.limit or None]
    if not paths:
        raise SystemExit("Nicio imagine de evaluat")
    # aceeasi limita pentru toate modurile, dar niciodata sub numarul de obiecte dintr-o imagine
    max_det = max(MAX_DET, *(len(read_labels(label_path(p))) for p in paths))
    for p in paths[:WARMUP]:
        detect(model, cv2.imread(str(p)), args)
    results, done, seconds, n_windows, areas = [], [], 0.0, [], []
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p}: imagine ilizibila, sarita")
            continue
        sync()
        start = time.perf_counter()
        det, windows = detect(model, img, args)
        sync()
        seconds += time.perf_counter() - start
        h, w = img.shape[:2]
        results.append((det[det[:, 4].argsort()[::-1][:max_det]], ground_truth(p, w, h)))
        done.append(p.name)
        n_windows.append(len(windows))
        areas.append(window_area(windows, w, h))
        if i % 50 == 0:
            print(f"{i}/{len(paths)}")
    map50, map50_95, per_class = mean_ap(results)
    names = YOLO(args.weights).names if args.mode == "sahi" else model.names
    report = {
        "mode": args.mode, "images": len(results), "conf": args.conf, "max_det": max_det,
        "map50_95": map50_95, "map50": map50,
        "ap_per_class": {names[c]: ap for c, ap in per_class.items()},
        "ms_per_image": 1000 * seconds / max(len(results), 1),
        "windows_per_image": float(np.mean(n_windows)), "window_area": float(np.mean(areas)),
    }
    print(f"mod {args.mode}, {len(results)} imagini, {report['ms_per_image']:.0f} ms/imagine ({WARMUP} de incalzire)")
    print(f"mAP50-95: {map50_95:.3f}  mAP50: {map50:.3f}")
    for name, ap in report["ap_per_class"].items():
        print(f"  {name:>16}: {ap:.3f}")
    if args.mode == "adaptive":
        print(f"ferestre: {report['windows_per_image']:.2f} pe imagine, pixeli procesati: {1 + report['window_area']:.2f}x imaginea")
        report["params"] = vars(args.params)
    if args.groups:
        groups = read_groups(args.groups)
        report["groups"] = {}
        for g in sorted(set(groups.values())):
            subset = [r for r, name in zip(results, done) if groups.get(name) == g]
            if subset:
                g50, g50_95, _ = mean_ap(subset)
                report["groups"][g] = {"images": len(subset), "map50_95": g50_95, "map50": g50}
                print(f"grupa {g} ({len(subset)} imagini): mAP50-95: {g50_95:.3f}  mAP50: {g50:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


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
    ap.add_argument("overrides", nargs="*", help="key=value pentru configs/adaptive.yaml (doar adaptive); se scriu imediat dupa weights")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--coco", action="store_true", help="model cu clase COCO: mapeaza etichetele VisDrone pe COCO")
    ap.add_argument("--mode", choices=["val", "full", "slice", "sahi", "adaptive"], default="val",
                    help="val = model.val() Ultralytics; full/slice/sahi/adaptive = mAP calculat de noi pe imaginea intreaga")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.001, help="acelasi prag mic pentru toate variantele")
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la combinarea detectiilor din ferestre")
    ap.add_argument("--slice", type=int, default=640, help="marimea ferestrei pentru slice si sahi")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--adaptive-config", default="configs/adaptive.yaml")
    ap.add_argument("--groups", type=Path, default=None, help="stats.csv de la pi-stats: mAP si pe fiecare grupa")
    ap.add_argument("--limit", type=int, default=0, help="doar primele N imagini (0 = toate)")
    ap.add_argument("--out", type=Path, default=None, help="fisier JSON cu rezultatele")
    args = ap.parse_args()
    if args.mode == "val":
        if args.limit or args.out or args.groups or args.overrides:
            raise SystemExit("--limit, --out, --groups si overrides nu se folosesc cu --mode val")
        validate(args)
        return
    if args.coco:
        raise SystemExit("--coco merge doar cu --mode val")
    if args.overrides and args.mode != "adaptive":
        raise SystemExit("overrides key=value se folosesc doar cu --mode adaptive")
    if args.mode == "adaptive":
        # --conf are prioritate fata de config, ca toate modurile sa fie evaluate la acelasi prag
        args.params = load_params(args.adaptive_config, (f"conf={args.conf}", *args.overrides))
    evaluate(args)


if __name__ == "__main__":
    main()
