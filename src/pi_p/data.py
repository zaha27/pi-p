"""Funcții comune pentru lucrul cu setul de date în format YOLO.

Toate celelalte module citesc și scriu datele prin funcțiile de aici, ca
formatul să fie definit într-un singur loc.

Structura unui set de date pe disc:

    datasets/VisDrone/
        images/train/0001.jpg      imaginea
        labels/train/0001.txt      etichetele ei, câte un obiect pe linie
        data.yaml                  fișierul pe care îl citește Ultralytics

O linie din fișierul de etichete are forma `clasă x y w h`, unde (x, y) este
centrul chenarului, iar (w, h) lățimea și înălțimea lui. Toate cele patru
valori sunt împărțite la dimensiunea imaginii, deci sunt între 0 și 1.
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# Cele 10 categorii din VisDrone, în ordinea folosită de model (clasa 0 = pedestrian).
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

# O culoare (R, G, B) pentru fiecare categorie, folosită la desenarea chenarelor.
COLORS = [(142, 85, 114), (212, 119, 138), (122, 139, 58), (42, 157, 143), (233, 162, 59),
          (200, 85, 61), (176, 141, 87), (92, 107, 115), (61, 90, 152), (74, 165, 192)]

# Folderul implicit al setului de date pregătit.
ROOT = Path("datasets/VisDrone")

IMG_EXT = {".jpg", ".jpeg", ".png"}


def images(root: Path, split: str) -> list[Path]:
    """Întoarce, sortate, toate imaginile dintr-un subset (train, val sau test)."""
    folder = root / "images" / split
    if not folder.is_dir():
        raise SystemExit(f"Lipsește {folder}, rulează întâi pi-prepare")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(img: Path) -> Path:
    """Calea fișierului de etichete al unei imagini.

    Exemplu: .../images/train/0001.jpg -> .../labels/train/0001.txt
    """
    dataset_root = img.parents[2]
    split = img.parent.name
    return dataset_root / "labels" / split / f"{img.stem}.txt"


def read_labels(path: Path) -> np.ndarray:
    """Citește etichetele ca tabel cu 5 coloane: clasă, x, y, w, h.

    O imagine fără obiecte are fișierul gol sau lipsă; atunci tabelul are 0 linii.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(path, ndmin=2, dtype=np.float32)


def write_labels(path: Path, labels: np.ndarray) -> None:
    """Scrie etichetele în format YOLO, câte un obiect pe linie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels]
    path.write_text("".join(lines))


def image_size(path: Path) -> tuple[int, int]:
    """Întoarce (lățime, înălțime) fără să decodeze toată imaginea."""
    with Image.open(path) as im:
        return im.size


def write_data_yaml(root: Path, splits: list[str]) -> Path:
    """Scrie data.yaml: unde sunt imaginile fiecărui subset și cum se numesc clasele."""
    cfg = {"path": str(root.resolve())}
    for split in splits:
        cfg[split] = f"images/{split}"
    cfg["names"] = dict(enumerate(NAMES))
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
