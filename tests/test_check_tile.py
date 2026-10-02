import numpy as np

from pi_p.check import check_label_file, check_split
from pi_p.data import NAMES, images, read_labels
from pi_p.prepare import process
from pi_p.tile import tile_image


def test_check_label_file_finds_issues(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 0.5 0.5 0.1 0.1\n12 0.5 0.5 0.1 0.1\n3 0.99 0.5 0.1 0.1\n3 0.5 0.5 0 0.1\n3 0.5\n3 0.5 0.5 0.1 0.1\n")
    kinds = [k for k, _ in check_label_file(p, len(NAMES))]
    assert kinds == ["clasă invalidă", "în afara imaginii", "arie zero", "format", "duplicat"]


def test_prepared_dataset_is_clean_and_tiles_are_valid(raw_visdrone, tmp_path):
    root = raw_visdrone.parent
    process(raw_visdrone, root, "train")
    assert not check_split(root, "train", len(NAMES))

    dst = tmp_path / "tiled"
    (dst / "images" / "train").mkdir(parents=True)
    n = sum(tile_image(p, dst, 320, 0.2, 0.5, 0.0, True) for p in images(root, "train"))
    tiles = images(dst, "train")
    assert n == len(tiles) > 2
    assert not check_split(dst, "train", len(NAMES))
    total = sum(len(read_labels(dst / "labels" / "train" / f"{t.stem}.txt")) for t in tiles if "_" in t.stem)
    assert total >= 5
