import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywhn2xyxy

from pi_p.adaptive import AdaptiveParams, adaptive_predict, detect, load_params
from pi_p.adaptive_figure import read_image
from pi_p.data import images, label_path, read_labels

WARMUP_IMAGES = 3
EVAL_CONF = 0.001
IOUV = torch.linspace(0.5, 0.95, 10)

Predictor = Callable[[np.ndarray], tuple[torch.Tensor, np.ndarray]]


class EvalResult(NamedTuple):
    metrics: DetMetrics
    images: int
    max_det: int
    ms_per_image: float
    windows_per_image: float
    window_area: float


def plain_predictor(model: YOLO, params: AdaptiveParams, device) -> Predictor:
    def predict(img: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        return detect(model, [img], params.prelim_imgsz, params.conf, device)[0], np.zeros((0, 4), np.int32)

    return predict


def adaptive_predictor(model: YOLO, params: AdaptiveParams, device) -> Predictor:
    def predict(img: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        return adaptive_predict(model, img, params, device)

    return predict


PREDICTORS = {"plain": plain_predictor, "adaptive": adaptive_predictor}


def val_images(data_yaml: str, limit: int | None) -> list[Path]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    paths = images(Path(cfg["path"]), Path(cfg["val"]).name)
    return paths[:limit] if limit else paths


def max_det_for(labels: list[np.ndarray]) -> int:
    return max([DEFAULT_CFG.max_det, *(len(lbl) for lbl in labels)])


def filter_classes(det: torch.Tensor, classes: list[int] | None) -> torch.Tensor:
    if classes is None:
        return det
    return det[torch.isin(det[:, 5], torch.tensor(classes, dtype=det.dtype))]


def cap_detections(det: torch.Tensor, max_det: int) -> torch.Tensor:
    return det[det[:, 4].argsort(descending=True)[:max_det]]


def match_image(det: torch.Tensor, gt_cls: torch.Tensor, gt_xyxy: torch.Tensor) -> np.ndarray:
    if len(det) == 0 or len(gt_cls) == 0:
        return np.zeros((len(det), len(IOUV)), dtype=bool)
    iou = box_iou(gt_xyxy, det[:, :4])
    # match_predictions folosește doar self.iouv, deci nu instanțiem un validator (ar crea foldere runs/)
    return BaseValidator.match_predictions(SimpleNamespace(iouv=IOUV), det[:, 5], gt_cls, iou).numpy()


def score_image(metrics: DetMetrics, det: torch.Tensor, labels: np.ndarray, width: int, height: int, name: str) -> None:
    gt_cls = torch.from_numpy(labels[:, 0])
    gt_xyxy = torch.from_numpy(xywhn2xyxy(labels[:, 1:], w=width, h=height))
    metrics.update_stats({
        "tp": match_image(det, gt_cls, gt_xyxy),
        "conf": det[:, 4].numpy(),
        "pred_cls": det[:, 5].numpy(),
        "target_cls": labels[:, 0],
        "target_img": np.unique(labels[:, 0]),
        "im_name": name,
    })


def window_area(windows: np.ndarray, width: int, height: int) -> float:
    w = windows.astype(np.int64)
    return float(((w[:, 2] - w[:, 0]) * (w[:, 3] - w[:, 1])).sum()) / (width * height)


def sync(device) -> None:
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()


def timed(predict: Predictor, img: np.ndarray, device) -> tuple[torch.Tensor, np.ndarray, float]:
    sync(device)
    start = time.perf_counter()
    det, windows = predict(img)
    sync(device)
    return det, windows, time.perf_counter() - start


def evaluate(predict: Predictor, paths: list[Path], names: dict[int, str], device, classes: list[int] | None) -> EvalResult:
    if not paths:
        raise SystemExit("Nicio imagine de evaluat")
    labels = [read_labels(label_path(p)) for p in paths]
    max_det = max_det_for(labels)
    for path in paths[:WARMUP_IMAGES]:
        predict(read_image(path))
    metrics = DetMetrics(names)
    seconds, n_windows, areas = [], [], []
    for path, lbl in zip(paths, labels):
        img = read_image(path)
        det, windows, elapsed = timed(predict, img, device)
        h, w = img.shape[:2]
        score_image(metrics, cap_detections(filter_classes(det, classes), max_det), lbl, w, h, path.name)
        seconds.append(elapsed)
        n_windows.append(len(windows))
        areas.append(window_area(windows, w, h))
    metrics.process()
    return EvalResult(metrics, len(paths), max_det, 1000 * float(np.mean(seconds)), float(np.mean(n_windows)), float(np.mean(areas)))


def report_lines(result: EvalResult, params: AdaptiveParams) -> list[str]:
    box, names = result.metrics.box, result.metrics.names
    lines = [f"mAP50-95: {box.map:.3f}  mAP50: {box.map50:.3f}"]
    lines += [f"  {names[c]:>16}: {box.maps[c]:.3f}" for c in box.ap_class_index]
    lines += [
        f"conf efectiv: {params.conf}  (prelim_conf {params.prelim_conf}, imgsz {params.prelim_imgsz}, max_det {result.max_det})",
        f"timp: {result.ms_per_image:.1f} ms/imagine ({result.images} imagini, {WARMUP_IMAGES} de încălzire)",
        f"ferestre: {result.windows_per_image:.2f} pe imagine",
        f"pixeli procesați: {1 + result.window_area:.2f}x imaginea (ferestre: {result.window_area:.2f}x)",
    ]
    return lines


def result_dict(result: EvalResult, mode: str, params: AdaptiveParams) -> dict:
    box, names = result.metrics.box, result.metrics.names
    return {
        "mode": mode,
        "images": result.images,
        "max_det": result.max_det,
        "conf": params.conf,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "ap_per_class": {names[c]: float(box.maps[c]) for c in box.ap_class_index},
        "ms_per_image": result.ms_per_image,
        "windows_per_image": result.windows_per_image,
        "window_area": result.window_area,
        "pixels_processed": 1 + result.window_area,
        "params": asdict(params),
    }


def run(model: YOLO, mode: str, data, config: str, overrides: tuple[str, ...], limit: int | None, device, out: Path | None, classes: list[int] | None) -> None:
    if mode not in PREDICTORS:
        raise SystemExit("--limit, --out și overrides se folosesc doar cu --mode plain sau adaptive")
    # conf=0.001 ca la model.val; un override explicit pus după el are prioritate (load le aplică în ordine)
    params = load_params(config, (f"conf={EVAL_CONF}", *overrides))
    paths = val_images(str(data), limit)
    result = evaluate(PREDICTORS[mode](model, params, device), paths, model.names, device, classes)
    print("\n".join(report_lines(result, params)))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result_dict(result, mode, params), indent=2, ensure_ascii=False))
