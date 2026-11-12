import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pi_p.config import load
from pi_p.predict import merge, to_image

MASK_ON = 255
CONNECTIVITY = 8


@dataclass(frozen=True)
class AdaptiveParams:
    mask_size: int
    dilate_kernel: int
    margin: float
    min_size: int
    prelim_imgsz: int
    prelim_conf: float
    window_imgsz: int
    conf: float
    iou: float


def load_params(path: str, overrides: tuple[str, ...] = ()) -> AdaptiveParams:
    return AdaptiveParams(**load(path, list(overrides)))


def boxes_to_mask(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        # floor/ceil acopera tot obiectul; max/min taie caseta la marginile imaginii
        left, top = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        right, bottom = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = MASK_ON
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    # kernel-ul par nu are pixel central, deci masca ar creste asimetric
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


def mask_to_rects(mask: np.ndarray) -> np.ndarray:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=CONNECTIVITY)
    # eticheta 0 e fundalul, nu o pata
    rects = stats[1:, :4].copy()
    rects[:, 2:] += rects[:, :2]
    return rects


def scale_rects(rects: np.ndarray, scale: float) -> np.ndarray:
    return rects.astype(np.float64) * scale


def grow_rects(rects: np.ndarray, margin: float) -> np.ndarray:
    return rects + np.array([-margin, -margin, margin, margin])


def enforce_min_size(rects: np.ndarray, min_size: float) -> np.ndarray:
    size = rects[:, 2:] - rects[:, :2]
    pad = np.maximum(min_size - size, 0) / 2
    return rects + np.hstack([-pad, pad])


def fit_in_image(rects: np.ndarray, width: int, height: int) -> np.ndarray:
    limit = np.array([width, height])
    shift = np.maximum(-rects[:, :2], 0) - np.maximum(rects[:, 2:] - limit, 0)
    moved = rects + np.hstack([shift, shift])
    return np.clip(moved, 0, np.hstack([limit, limit]))


def rects_to_windows(
    rects: np.ndarray, scale: float, margin: float, min_size: float, width: int, height: int
) -> np.ndarray:
    grown = grow_rects(scale_rects(rects, scale), margin)
    fitted = fit_in_image(enforce_min_size(grown, min_size), width, height)
    return np.hstack([np.floor(fitted[:, :2]), np.ceil(fitted[:, 2:])]).astype(np.int32)


class WindowStages(NamedTuple):
    scale: float
    mask: np.ndarray
    dilated: np.ndarray
    rects: np.ndarray
    windows: np.ndarray


class AdaptiveTrace(NamedTuple):
    prelim: torch.Tensor
    seeds: torch.Tensor
    stages: WindowStages
    detections: torch.Tensor


def window_stages(boxes_xyxy: np.ndarray, width: int, height: int, params: AdaptiveParams) -> WindowStages:
    # masca e o copie micsorata a imaginii (latura mare = mask_size), dar niciodata marita
    scale = max(max(width, height) / params.mask_size, 1.0)
    mask = boxes_to_mask(boxes_xyxy / scale, math.ceil(width / scale), math.ceil(height / scale))
    dilated = dilate_mask(mask, params.dilate_kernel)
    rects = mask_to_rects(dilated)
    windows = rects_to_windows(rects, scale, params.margin, params.min_size, width, height)
    return WindowStages(scale, mask, dilated, rects, windows)


def boxes_to_windows(boxes_xyxy: np.ndarray, width: int, height: int, params: AdaptiveParams) -> np.ndarray:
    return window_stages(boxes_xyxy, width, height, params).windows


def detect(model: YOLO, imgs: list[np.ndarray], imgsz: int, conf: float, device) -> list[torch.Tensor]:
    preds = model.predict(imgs, imgsz=imgsz, conf=conf, device=device, verbose=False)
    return [p.boxes.data.cpu() for p in preds]


def window_detections(model: YOLO, img: np.ndarray, windows: np.ndarray, params: AdaptiveParams, device) -> list[torch.Tensor]:
    h, w = img.shape[:2]
    wins = windows.tolist()
    crops = [np.ascontiguousarray(img[y1:y2, x1:x2]) for x1, y1, x2, y2 in wins]
    dets = detect(model, crops, params.window_imgsz, params.conf, device)
    return [to_image(det, tuple(win), w, h) for det, win in zip(dets, wins)]


def keep_confident(det: torch.Tensor, conf: float) -> torch.Tensor:
    return det[det[:, 4] >= conf]


def adaptive_trace(model: YOLO, img: np.ndarray, params: AdaptiveParams, device) -> AdaptiveTrace:
    h, w = img.shape[:2]

    # 1. o trecere pe toata imaginea, la pragul mai mic dintre cele doua, ca sa le serveasca pe amandoua
    prelim = detect(model, [img], params.prelim_imgsz, min(params.prelim_conf, params.conf), device)[0]

    # 2. din detectiile sigure se construiesc ferestrele
    seeds = keep_confident(prelim, params.prelim_conf)
    stages = window_stages(seeds[:, :4].numpy(), w, h, params)

    # 3. detectie in ferestre, apoi unirea cu detectiile de pe imaginea intreaga
    in_windows = window_detections(model, img, stages.windows, params, device) if len(stages.windows) else []
    whole = keep_confident(prelim, params.conf)
    detections = merge(torch.cat([whole, *in_windows]), params.iou)
    return AdaptiveTrace(prelim, seeds, stages, detections)


def adaptive_predict(model: YOLO, img: np.ndarray, params: AdaptiveParams, device) -> tuple[torch.Tensor, np.ndarray]:
    trace = adaptive_trace(model, img, params, device)
    return trace.detections, trace.stages.windows
