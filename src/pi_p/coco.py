import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.utils.ops import xywhn2xyxy

from pi_p.data import images, label_path, read_labels, write_labels
from pi_p.prepare import mask_ignored

TO_COCO = {0: 0, 1: 0, 2: 1, 3: 2, 5: 7, 8: 5, 9: 3}
ROOT = Path("datasets/VisDrone-coco")


def remap(img_path: Path, dst: Path) -> bool:
    split = img_path.parent.name
    lbl = read_labels(label_path(img_path))
    mapped = np.isin(lbl[:, 0], list(TO_COCO))
    out = lbl[mapped].copy()
    out[:, 0] = [TO_COCO[int(c)] for c in out[:, 0]]
    write_labels(dst / "labels" / split / f"{img_path.stem}.txt", out)
    target = dst / "images" / split / img_path.name
    if mapped.all():
        shutil.copy2(img_path, target)
        return False
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    xyxy = xywhn2xyxy(lbl[:, 1:], w=w, h=h).round().astype(int)
    mask_ignored(img, xyxy[mapped], xyxy[~mapped])
    cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def build(src: Path, split: str, names: dict[int, str], dst: Path = ROOT) -> Path:
    out = dst / "data.yaml"
    if out.is_file():
        return out
    (dst / "images" / split).mkdir(parents=True)
    masked = sum(remap(p, dst) for p in images(src, split))
    out.write_text(yaml.safe_dump({"path": str(dst.resolve()), "train": f"images/{split}", "val": f"images/{split}", "names": names}, sort_keys=False))
    print(f"{dst}: etichete mapate pe clasele COCO, {masked} imagini cu obiecte fără corespondent mascate")
    return out
