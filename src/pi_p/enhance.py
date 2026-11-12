import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import TQDM

from pi_p.data import ROOT, images, write_data_yaml
from pi_p.stats import read_groups

WEAK = {"dark", "low_contrast"}


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def needs_clahe(name: str, mode: str, groups: dict[str, str]) -> bool:
    return mode == "uniform" or groups.get(name) in WEAK


def enhance_image(src: Path, dst: Path, mode: str, groups: dict[str, str], clip: float, grid: int) -> bool:
    if not needs_clahe(src.name, mode, groups):
        shutil.copy2(src, dst / src.name)
        return False
    cv2.imwrite(str(dst / src.name), clahe(cv2.imread(str(src)), clip, grid), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uniform", "adaptive"], required=True)
    ap.add_argument("--src", type=Path, default=ROOT)
    ap.add_argument("--dst", type=Path, default=None, help="implicit datasets/VisDrone-clahe sau datasets/VisDrone-clahe-adaptive")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dst = args.dst or args.src.parent / ("VisDrone-clahe" if args.mode == "uniform" else "VisDrone-clahe-adaptive")
    if dst.exists():
        raise SystemExit(f"{dst} exista deja; sterge-l sau alege alt --dst")
    groups = read_groups(args.src / "stats.csv") if args.mode == "adaptive" else {}
    for split in args.splits:
        imgs = images(args.src, split)
        out = dst / "images" / split
        out.mkdir(parents=True)
        shutil.copytree(args.src / "labels" / split, dst / "labels" / split)
        fn = partial(enhance_image, dst=out, mode=args.mode, groups=groups, clip=args.clip, grid=args.grid)
        with ThreadPoolExecutor(args.workers) as pool:
            n = sum(TQDM(pool.map(fn, imgs), total=len(imgs), desc=split))
        print(f"{split}: CLAHE pe {n} din {len(imgs)} imagini")
    print(f"Dataset gata: {write_data_yaml(dst, args.splits)}")


if __name__ == "__main__":
    main()
