import cv2
import numpy as np
import yaml

from pi_p.coco import build
from pi_p.data import read_labels, write_labels
from pi_p.prepare import FILL


def test_build_maps_classes_and_masks_unmatched(tmp_path):
    src = tmp_path / "VisDrone"
    (src / "images" / "val").mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "val" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    cv2.imwrite(str(src / "images" / "val" / "b.jpg"), np.zeros((100, 200, 3), np.uint8))
    write_labels(src / "labels" / "val" / "a.txt", np.array([[1, 0.25, 0.5, 0.1, 0.2], [4, 0.75, 0.5, 0.1, 0.2]]))
    write_labels(src / "labels" / "val" / "b.txt", np.array([[9, 0.5, 0.5, 0.1, 0.2]]))
    dst = tmp_path / "coco"

    cfg = yaml.safe_load(build(src, "val", {0: "person"}, dst).read_text())

    assert cfg["val"] == "images/val"
    assert read_labels(dst / "labels" / "val" / "a.txt")[:, 0].tolist() == [0]
    assert read_labels(dst / "labels" / "val" / "b.txt")[:, 0].tolist() == [3]
    img = cv2.imread(str(dst / "images" / "val" / "a.jpg"))
    assert abs(int(img[50, 150].mean()) - FILL) < 5
    assert int(img[50, 50].mean()) < 5
