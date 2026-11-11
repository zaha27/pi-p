import cv2
import numpy as np

from pi_p.clahe import apply_clahe, convert_image, needs_clahe
from pi_p.data import write_labels
from pi_p.stats import classify, compute_thresholds, measure, read_groups, write_table

THRESHOLDS = {"luminozitate": 60.0, "contrast": 25.0}


def gradient(low: int, high: int) -> np.ndarray:
    row = np.linspace(low, high, 200).astype(np.uint8)
    return cv2.cvtColor(np.tile(row, (100, 1)), cv2.COLOR_GRAY2BGR)


def test_measure_separates_dark_flat_and_normal_images():
    dark = measure(gradient(0, 40))
    flat = measure(gradient(120, 140))
    normal = measure(gradient(0, 255))
    assert dark["luminozitate"] < flat["luminozitate"]
    assert flat["contrast"] < normal["contrast"]
    assert classify(dark, THRESHOLDS) == "intuneric"
    assert classify(flat, THRESHOLDS) == "contrast_slab"
    assert classify(normal, THRESHOLDS) == "normal"


def test_noise_estimate_grows_with_noise():
    rng = np.random.default_rng(0)
    clean = np.full((100, 100, 3), 128, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    assert measure(clean)["zgomot"] < 0.5 < measure(noisy)["zgomot"]


def test_thresholds_are_percentiles_of_train():
    rows = [{"luminozitate": float(v), "contrast": float(v) / 2} for v in range(101)]
    thr = compute_thresholds(rows, 15)
    assert np.isclose(thr["luminozitate"], 15) and np.isclose(thr["contrast"], 7.5)


def test_table_round_trip(tmp_path):
    rows = [{"imagine": "a.jpg", "luminozitate": 10.0, "contrast": 5.0, "zgomot": 1.0, "grupa": "intuneric"},
            {"imagine": "b.jpg", "luminozitate": 120.0, "contrast": 50.0, "zgomot": 1.0, "grupa": "normal"}]
    write_table(tmp_path / "stats" / "val.csv", rows)
    assert read_groups(tmp_path / "stats" / "val.csv") == {"a": "intuneric", "b": "normal"}


def test_clahe_raises_contrast_and_keeps_shape():
    img = gradient(100, 130)
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert measure(out)["contrast"] > measure(img)["contrast"]


def test_adaptive_mode_skips_normal_images(tmp_path):
    assert needs_clahe(gradient(0, 255), "uniform", None)
    assert not needs_clahe(gradient(0, 255), "adaptiv", THRESHOLDS)
    assert needs_clahe(gradient(0, 40), "adaptiv", THRESHOLDS)

    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "images" / "val").mkdir(parents=True)
    (dst / "images" / "val").mkdir(parents=True)
    (dst / "labels" / "val").mkdir(parents=True)
    for name, img in (("dark", gradient(0, 40)), ("normal", gradient(0, 255))):
        cv2.imwrite(str(src / "images" / "val" / f"{name}.png"), img)
        write_labels(src / "labels" / "val" / f"{name}.txt", np.array([[3, 0.5, 0.5, 0.1, 0.1]]))

    changed = {p.stem: convert_image(p, dst, "adaptiv", THRESHOLDS) for p in sorted((src / "images" / "val").iterdir())}
    assert changed == {"dark": True, "normal": False}
    assert (dst / "labels" / "val" / "dark.txt").read_text() == (src / "labels" / "val" / "dark.txt").read_text()
    same = cv2.imread(str(dst / "images" / "val" / "normal.png"))
    assert (same == gradient(0, 255)).all()
