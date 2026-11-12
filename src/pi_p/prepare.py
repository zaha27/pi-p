import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS_URL, TQDM
from ultralytics.utils.downloads import download
from ultralytics.utils.ops import xyxy2xywhn

from pi_p.data import ROOT, image_size, write_data_yaml, write_labels

RAW = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}

# Griul cu care acoperim regiunile ignorate. E aceeasi valoare cu care YOLO
# completeaza marginile imaginii, deci modelul o vede ca „nimic aici”.
FILL = 114


def parse(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append([int(v) for v in line.strip().split(",")[:8]])
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def convert(ann: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, bw, bh = ann[:, 0], ann[:, 1], ann[:, 2], ann[:, 3]
    score, category = ann[:, 4], ann[:, 5]

    # Din (colt, latime, inaltime) in (colt stanga-sus, colt dreapta-jos),
    # taiat la marginile imaginii.
    xyxy = np.stack([x, y, x + bw, y + bh], axis=1)
    xyxy = np.clip(xyxy, 0, [w, h, w, h])

    has_area = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    is_object = (score != 0) & (category >= 1) & (category <= 10)
    keep = has_area & is_object
    ignore = has_area & ~is_object

    # VisDrone numeroteaza clasele de la 1, YOLO de la 0.
    return category[keep] - 1, xyxy[keep], xyxy[ignore]


def mask_ignored(img: np.ndarray, keep: np.ndarray, ignore: np.ndarray) -> bool:
    mask = np.zeros(img.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in ignore:
        mask[y1:y2, x1:x2] = True
    # Un obiect adnotat care cade intr-o regiune ignorata trebuie sa ramana vizibil.
    for x1, y1, x2, y2 in keep:
        mask[y1:y2, x1:x2] = False
    img[mask] = FILL
    return bool(mask.any())


def process(raw: Path, root: Path, split: str) -> dict:
    img_dir = root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "objects": 0, "ignored_regions": 0, "masked_images": 0}

    for ann_file in TQDM(sorted((raw / "annotations").glob("*.txt")), desc=split):
        dst = img_dir / f"{ann_file.stem}.jpg"
        src = raw / "images" / dst.name
        if not src.exists():
            # Imaginea a fost deja mutata la o rulare anterioara, intrerupta.
            src = dst

        w, h = image_size(src)
        classes, keep, ignore = convert(parse(ann_file.read_text()), w, h)

        # Deschidem imaginea doar daca are regiuni de acoperit; altfel o mutam ca atare.
        img = cv2.imread(str(src)) if len(ignore) else None
        if img is not None and mask_ignored(img, keep, ignore):
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["masked_images"] += 1
            if src != dst:
                src.unlink()
        elif src != dst:
            shutil.move(src, dst)

        # Din pixeli (x1, y1, x2, y2) in formatul YOLO (centru si dimensiuni, intre 0 si 1).
        boxes = xyxy2xywhn(keep.astype(np.float64), w=w, h=h)
        write_labels(root / "labels" / split / f"{ann_file.stem}.txt", np.column_stack([classes, boxes]))

        stats["images"] += 1
        stats["objects"] += len(classes)
        stats["ignored_regions"] += len(ignore)

    # Pastram adnotarile originale, in caz ca vrem sa le recitim, si stergem restul.
    ann_dir = root / "annotations" / split
    ann_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(raw / "annotations", ann_dir)
    shutil.rmtree(raw)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Descarca VisDrone si il converteste in format YOLO")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--splits", nargs="+", default=list(RAW), choices=list(RAW))
    args = ap.parse_args()
    root = args.root

    # Un subset e de facut daca arhiva lui e dezarhivata pe disc sau daca nu are inca etichete.
    todo = [s for s in args.splits if (root / RAW[s]).is_dir() or not (root / "labels" / s).is_dir()]
    missing = [s for s in todo if not (root / RAW[s]).is_dir()]
    if missing:
        download([f"{ASSETS_URL}/{RAW[s]}.zip" for s in missing], dir=root, delete=True, threads=len(missing))

    for split in todo:
        print(split, process(root / RAW[split], root, split))
    print(f"Dataset gata: {write_data_yaml(root, args.splits)}")


if __name__ == "__main__":
    main()
