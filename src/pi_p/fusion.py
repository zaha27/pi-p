"""Fuziunea detecțiilor venite din mai multe ferestre.

Pentru că ferestrele se suprapun, apar două probleme:

  - un obiect tăiat de marginea unei ferestre e detectat ca „jumătate de obiect”;
  - un obiect din zona de suprapunere e detectat de două ori.

`drop_cut` rezolvă prima problemă, `merge` pe a doua.

O detecție este o linie cu 6 valori: x1, y1, x2, y2, scor, clasă.
Detalii și exemple în docs/pipeline/07-fuziunea-detectiilor.md.
"""

import torch

from pi_p.slicing import Window


def drop_cut(det: torch.Tensor, win: Window, w: int, h: int, margin: int = 2) -> torch.Tensor:
    """Elimină detecțiile tăiate de marginea ferestrei.

    `det` are coordonatele în fereastră (originea în colțul ei stânga-sus).
    O detecție e considerată tăiată dacă atinge, la cel mult `margin` pixeli,
    o margine a ferestrei aflată ÎN INTERIORUL imaginii. Marginile ferestrei
    care coincid cu marginea imaginii nu contează: acolo obiectul e tăiat de
    fotografie, nu de fereastră, și nu există o fereastră vecină care să îl
    vadă întreg.
    """
    x1, y1, x2, y2 = win
    win_w, win_h = x2 - x1, y2 - y1

    cut = torch.zeros(len(det), dtype=torch.bool)
    if x1 > 0:  # marginea stângă a ferestrei e în interiorul imaginii
        cut |= det[:, 0] <= margin
    if y1 > 0:  # marginea de sus
        cut |= det[:, 1] <= margin
    if x2 < w:  # marginea dreaptă
        cut |= det[:, 2] >= win_w - margin
    if y2 < h:  # marginea de jos
        cut |= det[:, 3] >= win_h - margin
    return det[~cut]


def intersection_over_smaller(boxes: torch.Tensor) -> torch.Tensor:
    """Matricea IOS: pentru fiecare pereche de chenare, intersecția împărțită la aria celui mai mic.

    Folosim IOS, nu IoU, pentru că prinde cazul „chenar mic aproape complet
    în interiorul unuia mare” (o bucată de mașină în interiorul mașinii
    întregi): IoU e mic acolo, IOS e aproape 1.
    """
    top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = torch.minimum(boxes[:, None, 2:4], boxes[None, :, 2:4])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    area = (boxes[:, 2:4] - boxes[:, :2]).prod(dim=1)
    smaller = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-6)
    return intersection / smaller


def merge(det: torch.Tensor, thr: float) -> torch.Tensor:
    """Elimină dublurile: păstrează, din fiecare grup de detecții suprapuse, pe cea cu scorul cel mai mare.

    Este o suprimare a non-maximelor (NMS) cu IOS în loc de IoU. Două detecții
    sunt dubluri dacă au aceeași clasă și IOS peste `thr`.
    """
    # Ordonăm descrescător după scor, ca detecțiile sigure să fie luate primele.
    det = det[det[:, 4].argsort(descending=True)]

    same_class = det[:, None, 5] == det[None, :, 5]
    duplicate = (intersection_over_smaller(det[:, :4]) > thr) & same_class

    keep = torch.ones(len(det), dtype=torch.bool)
    for i in range(len(det)):
        if keep[i]:
            # Detecția i rămâne; toate dublurile ei cu scor mai mic sunt eliminate.
            keep[i + 1 :] &= ~duplicate[i, i + 1 :]
    return det[keep]
