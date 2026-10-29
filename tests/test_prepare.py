import cv2
import numpy as np

from pi_p.data import read_labels
from pi_p.prepare import FILL, convert, mask_ignored, parse, process


def test_convert_drops_ignored_and_others_and_clips():
    ann = parse("100,100,50,40,1,4,0,0\n0,0,80,60,0,0,0,0\n500,300,30,30,1,11,0,0\n780,400,40,30,1,6,0,0\n")
    cls, keep, ignore = convert(ann, 800, 450)
    assert cls.tolist() == [3, 5]
    assert keep[1].tolist() == [780, 400, 800, 430]
    assert len(ignore) == 2


def test_mask_preserves_objects_inside_ignored_region():
    img = np.zeros((100, 100, 3), np.uint8)
    masked = mask_ignored(img, np.array([[10, 10, 20, 20]]), np.array([[0, 0, 50, 50]]))
    assert masked
    assert (img[0, 0] == FILL).all()
    assert (img[15, 15] == 0).all()
    assert (img[60, 60] == 0).all()


def test_process_end_to_end(raw_visdrone):
    root = raw_visdrone.parent
    stats = process(raw_visdrone, root, "train")
    assert stats == {"images": 2, "objects": 5, "ignored_regions": 2, "masked_images": 1}
    assert not raw_visdrone.exists()
    assert (root / "annotations" / "train" / "0001.txt").is_file()
    lbl = read_labels(root / "labels" / "train" / "0001.txt")
    assert lbl[:, 0].tolist() == [3, 0, 5]
    assert ((lbl[:, 1:] >= 0) & (lbl[:, 1:] <= 1)).all()
    img = cv2.imread(str(root / "images" / "train" / "0001.jpg"))
    assert abs(int(img[30, 40].mean()) - FILL) < 5
