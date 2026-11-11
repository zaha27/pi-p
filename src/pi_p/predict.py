"""Detecția pe o imagine: întreagă, pe grilă de ferestre sau pe ferestre adaptive.

Toate modurile întorc același lucru: un tabel de detecții în coordonatele
imaginii întregi, câte o linie `x1, y1, x2, y2, scor, clasă`.

Moduri (constanta MODES):

  - `intreg`:        modelul vede doar imaginea întreagă, micșorată;
  - `grila`:         imaginea întreagă + toate ferestrele grilei fixe;
  - `adaptiv`:       imaginea întreagă + ferestre doar unde sunt obiecte;
  - `adaptiv+grila`: ferestrele adaptive și grila, împreună.

`pi-predict` rulează un model pe imagini sau video și salvează rezultatul
desenat. Detalii în docs/pipeline/06-detectia-pe-ferestre.md.
"""

import argparse
from collections import Counter
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pi_p import adaptive
from pi_p.clahe import apply_clahe
from pi_p.data import IMG_EXT
from pi_p.fusion import drop_cut, merge
from pi_p.slicing import Window, windows

MODES = ["intreg", "grila", "adaptiv", "adaptiv+grila"]

# Câte detecții pe imagine acceptăm cel mult. Imaginile VisDrone au până la
# ~900 de obiecte, iar limita implicită din Ultralytics (300) ar pierde din ele.
MAX_DET = 1000

# Parametrii ferestrelor adaptive (pragul preliminar, dilatarea, marginea etc.).
ADAPTIVE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "adaptive.yaml"


def detect_whole(model: YOLO, img: np.ndarray, size: int, conf: float, device) -> torch.Tensor:
    """Rulează modelul pe imaginea întreagă, micșorată la `size` pe latura mare."""
    result = model.predict(img, imgsz=size, conf=conf, max_det=MAX_DET, device=device, verbose=False)[0]
    return result.boxes.data.cpu()


def detect_in_windows(model: YOLO, img: np.ndarray, wins: list[Window], size: int, conf: float, device) -> torch.Tensor:
    """Rulează modelul pe fiecare fereastră și mută detecțiile în coordonatele imaginii.

    Detecțiile tăiate de marginea ferestrei sunt eliminate aici (vezi fusion.drop_cut).
    """
    if not wins:
        return torch.zeros((0, 6))
    h, w = img.shape[:2]
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
    results = model.predict(crops, imgsz=size, conf=conf, max_det=MAX_DET, device=device, verbose=False)

    detections = []
    for result, win in zip(results, wins):
        det = drop_cut(result.boxes.data.cpu(), win, w, h)
        # Din coordonatele ferestrei în ale imaginii: adunăm colțul stânga-sus al ferestrei.
        x1, y1 = win[0], win[1]
        detections.append(det + torch.tensor([x1, y1, x1, y1, 0, 0]))
    return torch.cat(detections)


def sliced_predict(model: YOLO, img: np.ndarray, size: int, overlap: float, conf: float, iou: float, device) -> torch.Tensor:
    """Modul `grila`: imaginea întreagă + toate ferestrele grilei, apoi fuziune."""
    h, w = img.shape[:2]
    whole = detect_whole(model, img, size, conf, device)
    in_windows = detect_in_windows(model, img, windows(w, h, size, overlap), size, conf, device)
    return merge(torch.cat([whole, in_windows]), iou)


def adaptive_predict(model: YOLO, img: np.ndarray, size: int, overlap: float, conf: float, iou: float, device,
                     with_grid: bool = False) -> torch.Tensor:
    """Modurile `adaptiv` și `adaptiv+grila`.

    Ferestrele adaptive vin din `pi_p.adaptive`: detecție preliminară pe imaginea
    întreagă, mască, dilatare, câte o fereastră pe fiecare grup de obiecte.
    Parametrii lor sunt în configs/adaptive.yaml; `conf` și `iou` vin de aici,
    ca toate modurile să fie comparate cu același prag.
    """
    found, _ = adaptive.adaptive_predict(model, img, adaptive_params(conf, iou), device)
    if not with_grid:
        return found

    # Grila rămâne ca plasă de siguranță pentru obiectele ratate de detecția preliminară.
    h, w = img.shape[:2]
    in_grid = detect_in_windows(model, img, windows(w, h, size, overlap), size, conf, device)
    return merge(torch.cat([found, in_grid]), iou)


@cache
def adaptive_params(conf: float, iou: float) -> adaptive.AdaptiveParams:
    """Parametrii din configs/adaptive.yaml, cu pragurile modului curent."""
    return adaptive.load_params(str(ADAPTIVE_CONFIG), (f"conf={conf}", f"iou={iou}"))


def detect(model: YOLO, img: np.ndarray, mode: str, size: int, overlap: float, conf: float, iou: float, device) -> torch.Tensor:
    """Rulează detecția în modul cerut. Punctul comun de intrare pentru predicție și evaluare."""
    if mode == "intreg":
        return detect_whole(model, img, size, conf, device)
    if mode == "grila":
        return sliced_predict(model, img, size, overlap, conf, iou, device)
    if mode in ("adaptiv", "adaptiv+grila"):
        return adaptive_predict(model, img, size, overlap, conf, iou, device, with_grid=mode == "adaptiv+grila")
    raise ValueError(f"Mod necunoscut: {mode}; alege din {MODES}")


def image_paths(source: Path) -> list[Path]:
    """Imaginile de procesat: un fișier sau toate imaginile dintr-un folder."""
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
    if source.is_file() and source.suffix.lower() in IMG_EXT:
        return [source]
    raise SystemExit(f"--slice merge doar pe imagini sau foldere cu imagini: {source}")


def report(r: Results) -> None:
    """Afișează câte obiecte din fiecare clasă au fost găsite într-o imagine."""
    print(f"{r.path}: {dict(Counter(r.names[int(c)] for c in r.boxes.cls))}")


def main():
    ap = argparse.ArgumentParser(description="Rulează un model pe imagini sau video și salvează detecțiile desenate")
    ap.add_argument("weights")
    ap.add_argument("source", help="imagine, folder, video sau URL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la combinarea detecțiilor din ferestre")
    ap.add_argument("--device", default=None)
    ap.add_argument("--slice", type=int, default=0, help="mărimea ferestrei pentru inferență pe bucăți (0 = dezactivat)")
    ap.add_argument("--windows", choices=MODES[1:], default="grila", help="cum sunt alese ferestrele, când --slice e activ")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--clahe", action="store_true", help="aplică CLAHE pe imagine înainte de detecție")
    ap.add_argument("--out", type=Path, default=Path("runs/predict"))
    args = ap.parse_args()
    model = YOLO(args.weights)

    # Fără ferestre lăsăm Ultralytics să facă tot: știe și video, și URL-uri.
    if not args.slice and not args.clahe:
        for r in model.predict(args.source, conf=args.conf, device=args.device, save=True, stream=True,
                               project=args.out.parent, name=args.out.name):
            report(r)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    mode = args.windows if args.slice else "intreg"
    size = args.slice or 640
    for path in image_paths(Path(args.source)):
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path}: imagine ilizibilă, sărită")
            continue
        if args.clahe:
            img = apply_clahe(img)
        boxes = detect(model, img, mode, size, args.overlap, args.conf, args.iou, args.device)
        r = Results(img, path=str(path), names=model.names, boxes=boxes)
        cv2.imwrite(str(args.out / path.name), r.plot(line_width=1, labels=False))
        report(r)


if __name__ == "__main__":
    main()
