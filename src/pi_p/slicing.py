"""Felierea pe grilă fixă: împarte o imagine în ferestre pătrate suprapuse.

Aceleași funcții sunt folosite în două locuri:
  - la antrenare, de `pi-tile`, care taie imaginile de train în felii;
  - la inferență, de `predict.py`, care rulează modelul pe fiecare fereastră.

O fereastră este un tuplu (x1, y1, x2, y2) în pixelii imaginii întregi.
Detalii și desene în docs/pipeline/04-felierea-pe-grila.md.
"""

import numpy as np

Window = tuple[int, int, int, int]


def starts(length: int, size: int, step: int) -> list[int]:
    """Pozițiile de început ale ferestrelor de-a lungul unei laturi.

    Ferestrele pornesc din `step` în `step`, iar ultima e lipită de marginea
    imaginii, ca să nu rămână o fâșie neacoperită.

    Exemplu: latură de 1360, fereastră de 640, pas 512 -> [0, 512, 720].
    Ultima fereastră începe la 720 (= 1360 - 640), nu la 1024.
    """
    if length <= size:
        return [0]
    return [*range(0, length - size, step), length - size]


def windows(w: int, h: int, size: int, overlap: float) -> list[Window]:
    """Toate ferestrele care acoperă o imagine de `w` x `h` pixeli.

    `overlap` este fracția cu care se suprapun două ferestre vecine: la 0.2 și
    ferestre de 640, pasul este 512, deci vecinele au în comun o bandă de 128 px.
    Suprapunerea face ca un obiect tăiat de o fereastră să apară întreg în vecina ei.
    """
    step = max(1, round(size * (1 - overlap)))
    wins = []
    for y in starts(h, size, step):
        for x in starts(w, size, step):
            wins.append((x, y, min(x + size, w), min(y + size, h)))
    return wins


def crop_boxes(xyxy: np.ndarray, win: Window, min_visibility: float) -> tuple[np.ndarray, np.ndarray]:
    """Mută chenarele unei imagini în coordonatele unei ferestre.

    Un chenar e păstrat doar dacă în fereastră rămâne vizibilă cel puțin
    fracția `min_visibility` din aria lui. Altfel am învăța modelul să
    detecteze „o bucată de mașină”.

    Întoarce (keep, boxes): `keep` spune care chenare au fost păstrate, iar
    `boxes` sunt chenarele păstrate, tăiate la fereastră și cu originea în
    colțul ei stânga-sus.
    """
    x1, y1, x2, y2 = win

    # Partea din fiecare chenar care cade în fereastră.
    clipped = xyxy.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(x1, x2)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(y1, y2)

    full_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    visible_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    keep = (visible_area > 0) & (visible_area >= min_visibility * full_area)

    origin = np.array([x1, y1, x1, y1], dtype=clipped.dtype)
    return keep, clipped[keep] - origin
