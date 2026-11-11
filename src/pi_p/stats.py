"""Analiza statistică a imaginilor: cât de luminoasă și de contrastată e fiecare.

Pentru fiecare imagine măsurăm trei indicatori pe canalul de luminanță:

  - luminozitatea: media intensităților (0 = negru, 255 = alb);
  - contrastul: abaterea standard a intensităților;
  - zgomotul: o estimare a zgomotului din imagine (metoda Immerkær).

Pe baza lor, imaginea intră într-una din trei grupe:

  - `intuneric`: luminozitatea e sub pragul de luminozitate;
  - `contrast_slab`: nu e întunecată, dar contrastul e sub pragul de contrast;
  - `normal`: restul.

Pragurile nu sunt alese „din ochi”: sunt percentile ale indicatorilor pe
subsetul train (implicit percentila 15). Aceleași praguri se folosesc apoi
pentru val și test.

`pi-stats` scrie, în `<root>/stats/`, câte un tabel CSV pe subset și fișierul
`praguri.json`. Grupele sunt folosite de CLAHE adaptiv și de evaluarea pe
grupe. Detalii în docs/pipeline/02-analiza-statistica.md.
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import TQDM

from pi_p.data import ROOT, images

GROUPS = ["intuneric", "contrast_slab", "normal"]
COLUMNS = ["imagine", "luminozitate", "contrast", "zgomot", "grupa"]

# Nucleul din metoda Immerkær: răspunde puternic la zgomot și aproape deloc
# la muchii și la zone netede.
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Canalul L (luminanță) al unei imagini BGR, cu valori între 0 și 255."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]


def estimate_noise(lum: np.ndarray) -> float:
    """Estimează abaterea standard a zgomotului (Immerkær, 1996)."""
    h, w = lum.shape
    response = cv2.filter2D(lum.astype(np.float32), -1, NOISE_KERNEL)
    # Ignorăm bordura de un pixel, unde filtrul nu are vecini compleți.
    total = np.abs(response[1:-1, 1:-1]).sum()
    return float(np.sqrt(np.pi / 2) * total / (6 * (w - 2) * (h - 2)))


def measure(img: np.ndarray) -> dict[str, float]:
    """Calculează cei trei indicatori ai unei imagini."""
    lum = luminance(img)
    return {
        "luminozitate": float(lum.mean()),
        "contrast": float(lum.std()),
        "zgomot": estimate_noise(lum),
    }


def compute_thresholds(rows: list[dict], percentile: float) -> dict[str, float]:
    """Alege pragurile ca percentile ale indicatorilor pe imaginile date (train)."""
    return {
        "luminozitate": float(np.percentile([r["luminozitate"] for r in rows], percentile)),
        "contrast": float(np.percentile([r["contrast"] for r in rows], percentile)),
        "percentila": percentile,
    }


def classify(indicators: dict[str, float], thresholds: dict[str, float]) -> str:
    """Încadrează o imagine într-o grupă, pe baza indicatorilor ei."""
    if indicators["luminozitate"] < thresholds["luminozitate"]:
        return "intuneric"
    if indicators["contrast"] < thresholds["contrast"]:
        return "contrast_slab"
    return "normal"


def measure_file(path: Path) -> dict:
    """Citește o imagine de pe disc și îi calculează indicatorii."""
    return {"imagine": path.name, **measure(cv2.imread(str(path)))}


def measure_split(root: Path, split: str, workers: int) -> list[dict]:
    """Calculează indicatorii pentru toate imaginile unui subset."""
    imgs = images(root, split)
    with ThreadPoolExecutor(workers) as pool:
        return list(TQDM(pool.map(measure_file, imgs), total=len(imgs), desc=split))


def write_table(path: Path, rows: list[dict]) -> None:
    """Scrie tabelul unui subset: o linie pe imagine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})


def read_groups(path: Path) -> dict[str, str]:
    """Citește un tabel scris de `write_table`; întoarce grupa fiecărei imagini.

    Cheia este numele fără extensie (ex. „0000001_02999_d_0000005”).
    """
    with open(path, newline="", encoding="utf-8") as f:
        return {Path(row["imagine"]).stem: row["grupa"] for row in csv.DictReader(f)}


def load_thresholds(root: Path) -> dict[str, float]:
    """Citește pragurile calculate pe train de o rulare anterioară a `pi-stats`."""
    path = root / "stats" / "praguri.json"
    if not path.is_file():
        raise SystemExit(f"Lipsește {path}; rulează întâi pi-stats (cu subsetul train)")
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description="Măsoară luminozitatea și contrastul imaginilor și le împarte pe grupe")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--percentile", type=float, default=15, help="percentila din train folosită ca prag")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = args.root / "stats"
    measured = {split: measure_split(args.root, split, args.workers) for split in args.splits}

    # Pragurile se calculează doar pe train; val și test le refolosesc.
    if "train" in measured:
        thresholds = compute_thresholds(measured["train"], args.percentile)
        out.mkdir(parents=True, exist_ok=True)
        (out / "praguri.json").write_text(json.dumps(thresholds, indent=2))
    else:
        thresholds = load_thresholds(args.root)
    print(f"Praguri: luminozitate < {thresholds['luminozitate']:.1f} = întuneric, "
          f"contrast < {thresholds['contrast']:.1f} = contrast slab")

    for split, rows in measured.items():
        for row in rows:
            row["grupa"] = classify(row, thresholds)
        write_table(out / f"{split}.csv", rows)
        counts = {g: sum(r["grupa"] == g for r in rows) for g in GROUPS}
        print(f"{split}: {counts} -> {out / f'{split}.csv'}")


if __name__ == "__main__":
    main()
