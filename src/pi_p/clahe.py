"""Îmbunătățirea contrastului cu CLAHE.

CLAHE (Contrast Limited Adaptive Histogram Equalization) împarte imaginea
într-o grilă de regiuni și egalizează histograma fiecăreia separat, cu o
limită care împiedică amplificarea zgomotului. Îl aplicăm doar pe canalul de
luminanță, ca să nu schimbăm culorile.

`pi-clahe` scrie un set de date nou, cu aceleași etichete, în care imaginile
au trecut prin CLAHE. Există două moduri:

  - `uniform`: CLAHE pe toate imaginile;
  - `adaptiv`: CLAHE doar pe imaginile din grupele `intuneric` și
    `contrast_slab` (vezi stats.py); imaginile normale sunt copiate neatinse.

CLAHE se aplică pe imaginea întreagă, înainte de feliere, la fel pentru
train, val și test. Detalii în docs/pipeline/03-imbunatatirea-contrastului.md.
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import TQDM

from pi_p.data import ROOT, images, label_path, write_data_yaml
from pi_p.stats import classify, load_thresholds, measure

# Valori uzuale; nu le optimizăm, pentru că fiecare încercare ar cere o reantrenare.
CLIP_LIMIT = 2.0
GRID = 8


def apply_clahe(img: np.ndarray, clip_limit: float = CLIP_LIMIT, grid: int = GRID) -> np.ndarray:
    """Aplică CLAHE pe luminanța unei imagini BGR și întoarce imaginea rezultată."""
    # 1. Trecem în spațiul Lab, unde L este luminanța, iar a și b sunt culoarea.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    # 2. Egalizăm doar luminanța.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lum = clahe.apply(lum)
    # 3. Punem canalele la loc și ne întoarcem în BGR.
    return cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2BGR)


def needs_clahe(img: np.ndarray, mode: str, thresholds: dict[str, float] | None) -> bool:
    """Decide dacă o imagine primește CLAHE, în funcție de mod."""
    if mode == "uniform":
        return True
    return classify(measure(img), thresholds) != "normal"


def convert_image(img_path: Path, dst: Path, mode: str, thresholds: dict[str, float] | None) -> bool:
    """Scrie în `dst` varianta unei imagini (cu sau fără CLAHE) și etichetele ei.

    Întoarce True dacă imaginea a primit CLAHE.
    """
    split = img_path.parent.name
    out = dst / "images" / split / img_path.name
    shutil.copy2(label_path(img_path), dst / "labels" / split / f"{img_path.stem}.txt")

    img = cv2.imread(str(img_path))
    if needs_clahe(img, mode, thresholds):
        cv2.imwrite(str(out), apply_clahe(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    shutil.copy2(img_path, out)
    return False


def main():
    ap = argparse.ArgumentParser(description="Scrie un set de date cu CLAHE aplicat pe imagini")
    ap.add_argument("mode", choices=["uniform", "adaptiv"])
    ap.add_argument("--src", type=Path, default=ROOT)
    ap.add_argument("--dst", type=Path, default=None, help="implicit datasets/VisDrone-clahe-<mod>")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dst = args.dst or args.src.with_name(f"{args.src.name}-clahe-{args.mode}")
    if dst.exists():
        raise SystemExit(f"{dst} există deja; șterge-l sau alege alt --dst")
    # Modul adaptiv are nevoie de pragurile calculate de pi-stats pe train.
    thresholds = load_thresholds(args.src) if args.mode == "adaptiv" else None

    convert = partial(convert_image, dst=dst, mode=args.mode, thresholds=thresholds)
    for split in args.splits:
        imgs = images(args.src, split)
        (dst / "images" / split).mkdir(parents=True)
        (dst / "labels" / split).mkdir(parents=True)
        with ThreadPoolExecutor(args.workers) as pool:
            enhanced = sum(TQDM(pool.map(convert, imgs), total=len(imgs), desc=split))
        print(f"{split}: CLAHE pe {enhanced} din {len(imgs)} imagini")
    print(f"Dataset gata: {write_data_yaml(dst, args.splits)}")


if __name__ == "__main__":
    main()
