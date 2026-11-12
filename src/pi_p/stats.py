"""Luminozitatea, contrastul și grupa de iluminare a fiecărei imagini (`pi-stats`)."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils import TQDM

from pi_p.data import ROOT, images

DARK = 70
LOW_CONTRAST = 35


def brightness_contrast(img: np.ndarray) -> tuple[float, float]:
    """Media și abaterea standard a canalului L (luminanța) din spațiul Lab."""
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    return float(l.mean()), float(l.std())


def group(brightness: float, contrast: float, dark: float = DARK, low_contrast: float = LOW_CONTRAST) -> str:
    """Regula de încadrare: întâi luminozitatea, apoi contrastul."""
    if brightness < dark:
        return "dark"
    if contrast < low_contrast:
        return "low_contrast"
    return "normal"


def measure(path: Path) -> tuple[float, float]:
    """Luminozitatea și contrastul unei imagini de pe disc."""
    # imaginea micșorată de 4 ori ajunge pentru medie și abatere și se citește mult mai repede
    return brightness_contrast(cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4))


def read_groups(path: Path) -> dict[str, str]:
    """Citește stats.csv: numele imaginii -> grupa ei."""
    if not path.is_file():
        raise SystemExit(f"Lipsește {path}, rulează întâi pi-stats")
    with open(path, newline="") as f:
        return {row["image"]: row["group"] for row in csv.DictReader(f)}


def plot(rows: list[dict], dark: float, low_contrast: float, out: Path) -> None:
    """Histogramele luminozității și contrastului, cu pragurile desenate."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    b = np.array([r["brightness"] for r in rows])
    c = np.array([r["contrast"] for r in rows])
    ax[0].hist(b, bins=64)
    ax[0].axvline(dark, color="r", ls="--")
    ax[0].set_title("Luminozitate medie (L)")
    ax[1].hist(c, bins=64)
    ax[1].axvline(low_contrast, color="r", ls="--")
    ax[1].set_title("Contrast (abaterea standard a lui L)")
    ax[2].scatter(b, c, s=2, alpha=0.3)
    ax[2].axvline(dark, color="r", ls="--")
    ax[2].axhline(low_contrast, color="r", ls="--")
    ax[2].set_xlabel("luminozitate")
    ax[2].set_ylabel("contrast")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--dark", type=float, default=DARK, help="sub această luminozitate imaginea e întunecată")
    ap.add_argument("--low-contrast", type=float, default=LOW_CONTRAST, help="sub acest contrast imaginea e slab contrastată")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for split in args.splits:
        imgs = images(args.root, split)
        with ThreadPoolExecutor(args.workers) as pool:
            values = list(TQDM(pool.map(measure, imgs), total=len(imgs), desc=split))
        for p, (b, c) in zip(imgs, values):
            rows.append({"split": split, "image": p.name, "brightness": round(b, 2), "contrast": round(c, 2),
                         "group": group(b, c, args.dark, args.low_contrast)})
        print(split, dict(Counter(r["group"] for r in rows if r["split"] == split)))
    with open(args.root / "stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot([r for r in rows if r["split"] == "train"], args.dark, args.low_contrast, args.root / "stats.png")
    print(f"Statistici în {args.root / 'stats.csv'}, histograme în {args.root / 'stats.png'}")


if __name__ == "__main__":
    main()
