"""Potrivirea detecțiilor cu etichetele și calculul mAP.

O detecție e corectă dacă are aceeași clasă cu un obiect adnotat și IoU-ul lor
trece pragul; fiecare obiect poate fi folosit o singură dată. AP-ul pe clasă
este calculat de Ultralytics (`ap_per_class`), iar mAP este media pe clase.
Detalii în docs/pipeline/08-evaluarea.md.
"""

import numpy as np
from ultralytics.utils.metrics import ap_per_class

THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU între fiecare chenar din `a` și fiecare chenar din `b` (matrice len(a) x len(b))."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(min=0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pentru fiecare detecție și fiecare prag IoU (0,50 ... 0,95): este corectă?"""
    # det: x1 y1 x2 y2 conf cls, sortat descrescător după conf; gt: cls x1 y1 x2 y2
    correct = np.zeros((len(det), len(THRESHOLDS)), dtype=bool)
    if len(det) == 0 or len(gt) == 0:
        return correct
    iou = box_iou(det[:, :4], gt[:, 1:])
    iou[det[:, 5][:, None] != gt[:, 0][None, :]] = 0
    for t, thr in enumerate(THRESHOLDS):
        used = np.zeros(len(gt), dtype=bool)
        for i in range(len(det)):
            free = np.where(~used & (iou[i] >= thr))[0]
            if len(free):
                j = free[iou[i, free].argmax()]
                used[j] = True
                correct[i, t] = True
    return correct


def mean_ap(images: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, dict[int, float]]:
    """mAP50, mAP50-95 și AP pe fiecare clasă, din perechile (detecții, etichete) ale tuturor imaginilor."""
    corrects, confs, pred_cls, target_cls = [], [], [], []
    for det, gt in images:
        det = det[det[:, 4].argsort()[::-1]]
        corrects.append(match(det, gt))
        confs.append(det[:, 4])
        pred_cls.append(det[:, 5])
        target_cls.append(gt[:, 0])
    conf = np.concatenate(confs)
    if len(conf) == 0:
        return 0.0, 0.0, {}
    ap, classes = ap_per_class(np.concatenate(corrects), conf, np.concatenate(pred_cls), np.concatenate(target_cls))[5:7]
    return float(ap[:, 0].mean()), float(ap.mean()), {int(c): float(a) for c, a in zip(classes, ap.mean(1))}
