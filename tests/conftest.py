import cv2
import numpy as np
import pytest

ANNOTATIONS = {
    "0001": [
        "100,100,50,40,1,4,0,0",
        "300,200,20,20,1,1,0,1",
        "0,0,80,60,0,0,0,0",
        "500,300,30,30,1,11,0,0",
        "780,400,40,30,1,6,0,0",
    ],
    "0002": [
        "10,10,30,30,1,10,0,0",
        "400,250,100,80,1,9,0,0",
    ],
}


@pytest.fixture
def raw_visdrone(tmp_path):
    raw = tmp_path / "VisDrone" / "VisDrone2019-DET-train"
    (raw / "images").mkdir(parents=True)
    (raw / "annotations").mkdir()
    rng = np.random.default_rng(0)
    for stem, rows in ANNOTATIONS.items():
        cv2.imwrite(str(raw / "images" / f"{stem}.jpg"), rng.integers(0, 255, (450, 800, 3), dtype=np.uint8))
        (raw / "annotations" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
    return raw
