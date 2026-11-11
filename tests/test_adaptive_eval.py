import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from pi_p import evaluate as evaluate_cli
from pi_p.adaptive_eval import cap_detections, evaluate, max_det_for, run, val_images
from pi_p.data import write_labels

NAMES = {i: f"class{i}" for i in range(10)}
# nume -> (lățime, înălțime, [(clasă, x1, y1, x2, y2), ...]); obiectele din "a" sunt depărtate între ele
IMAGES = {
    "a": (200, 100, [(0, 10, 10, 40, 40), (3, 150, 50, 190, 90)]),
    "b": (160, 120, [(3, 60, 40, 100, 80)]),
}
ORACLE = {
    (h, w): torch.tensor([[x1, y1, x2, y2, 0.9, c] for c, x1, y1, x2, y2 in boxes], dtype=torch.float32)
    for w, h, boxes in IMAGES.values()
}
HAND_WINDOWS = {(100, 200): [[0, 0, 100, 100]], (120, 160): [[0, 0, 80, 60], [80, 60, 160, 120]]}
NO_WINDOWS = np.zeros((0, 4), np.int32)
# interpolarea în 101 puncte din ultralytics (compute_ap) dă 0.995, nu 1.0, chiar și pentru predicții perfecte
PERFECT_AP = 0.995
CONFIG = "mask_size: 640\ndilate_kernel: 3\nmargin: 0\nmin_size: 640\nprelim_imgsz: 640\nprelim_conf: 0.25\nwindow_imgsz: 640\nconf: 0.25\niou: 0.5\n"


class FakeModel:
    names = NAMES

    def __init__(self):
        self.calls = []

    def predict(self, source, imgsz, conf, device, verbose):
        self.calls.append((imgsz, conf))
        return [SimpleNamespace(boxes=SimpleNamespace(data=ORACLE[img.shape[:2]].clone())) for img in source]


@pytest.fixture
def val(tmp_path):
    root = tmp_path / "ds"
    for name, (w, h, boxes) in IMAGES.items():
        (root / "images" / "val").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / "val" / f"{name}.jpg"), np.zeros((h, w, 3), np.uint8))
        labels = np.array([[c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h] for c, x1, y1, x2, y2 in boxes])
        write_labels(root / "labels" / "val" / f"{name}.txt", labels)
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "val": "images/val", "names": NAMES}))
    return data


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "adaptive.yaml"
    path.write_text(CONFIG)
    return path


def oracle_predictor(img):
    return ORACLE[img.shape[:2]].clone(), NO_WINDOWS


def silent_predictor(img):
    return torch.zeros((0, 6)), NO_WINDOWS


def hand_windows_predictor(img):
    return ORACLE[img.shape[:2]].clone(), np.array(HAND_WINDOWS[img.shape[:2]], np.int32)


def junk_predictor(img):
    junk = torch.tensor([[0, 0, 5, 5, 0.99, 7]], dtype=torch.float32).repeat(301, 1)
    return torch.cat([ORACLE[img.shape[:2]], junk]), NO_WINDOWS


def test_evaluate_perfect_predictions_give_perfect_map(val):
    result = evaluate(oracle_predictor, val_images(str(val), None), NAMES, "cpu", None)
    box = result.metrics.box
    assert box.map == pytest.approx(PERFECT_AP)
    assert box.map50 == pytest.approx(PERFECT_AP)
    assert [box.maps[c] for c in box.ap_class_index] == pytest.approx([PERFECT_AP, PERFECT_AP])
    assert list(box.ap_class_index) == [0, 3]


def test_evaluate_without_predictions_gives_map_zero(val):
    result = evaluate(silent_predictor, val_images(str(val), None), NAMES, "cpu", None)
    assert result.metrics.box.map == pytest.approx(0.0)


def test_evaluate_counts_windows_and_pixel_fraction_by_hand(val):
    result = evaluate(hand_windows_predictor, val_images(str(val), None), NAMES, "cpu", None)
    assert result.windows_per_image == pytest.approx(1.5)
    assert result.window_area == pytest.approx(0.5)


def test_val_images_respects_limit(val):
    assert [p.name for p in val_images(str(val), 1)] == ["a.jpg"]
    assert [p.name for p in val_images(str(val), None)] == ["a.jpg", "b.jpg"]


def test_cap_detections_keeps_highest_scores():
    det = torch.tensor([[0, 0, 1, 1, 0.2, 0], [0, 0, 1, 1, 0.9, 1], [0, 0, 1, 1, 0.5, 2]])
    assert cap_detections(det, 2)[:, 4].tolist() == pytest.approx([0.9, 0.5])


def test_max_det_for_follows_dataset_maximum():
    assert max_det_for([np.zeros((5, 5))]) == 300
    assert max_det_for([np.zeros((5, 5)), np.zeros((350, 5))]) == 350


def test_evaluate_classes_filter_runs_before_cap(val):
    paths = val_images(str(val), None)
    # 301 de casete cu scor mare într-o clasă exclusă: fără filtru ocupă tot plafonul de 300
    kept = evaluate(junk_predictor, paths, NAMES, "cpu", [0, 3])
    crowded = evaluate(junk_predictor, paths, NAMES, "cpu", None)
    assert kept.metrics.box.map == pytest.approx(PERFECT_AP)
    assert crowded.metrics.box.map == pytest.approx(0.0)


def test_run_adaptive_end_to_end_with_fake_model(val, cfg, tmp_path):
    out = tmp_path / "out" / "result.json"
    run(FakeModel(), "adaptive", val, str(cfg), (), None, "cpu", out, None)
    result = json.loads(out.read_text())
    assert result["images"] == 2
    assert result["map50_95"] == pytest.approx(PERFECT_AP)
    assert result["windows_per_image"] == pytest.approx(1.5)
    assert result["window_area"] == pytest.approx(1.5)
    assert result["pixels_processed"] == pytest.approx(2.5)


@pytest.mark.parametrize("overrides, expected", [((), 0.001), (("conf=0.05",), 0.05)], ids=["default_conf", "explicit_override"])
def test_run_plain_uses_eval_conf_unless_overridden(val, cfg, tmp_path, capsys, overrides, expected):
    model, out = FakeModel(), tmp_path / "result.json"
    run(model, "plain", val, str(cfg), overrides, None, "cpu", out, None)
    result = json.loads(out.read_text())
    assert set(model.calls) == {(640, expected)}
    assert result["conf"] == expected
    assert result["windows_per_image"] == 0
    assert result["pixels_processed"] == pytest.approx(1.0)
    assert f"conf efectiv: {expected}" in capsys.readouterr().out


def test_evaluate_main_plain_mode_prints_report_without_model_val(val, cfg, tmp_path, monkeypatch, capsys):
    out = tmp_path / "result.json"
    argv = ["pi-eval", "w.pt", "conf=0.01", "--mode", "plain", "--data", str(val), "--adaptive-config", str(cfg), "--limit", "1", "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(evaluate_cli, "YOLO", lambda weights: FakeModel())
    evaluate_cli.main()
    assert f"mAP50-95: {PERFECT_AP:.3f}" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["images"] == 1
    assert result["conf"] == 0.01


def test_evaluate_main_standard_mode_rejects_limit(val, monkeypatch):
    monkeypatch.setattr("sys.argv", ["pi-eval", "w.pt", "--data", str(val), "--limit", "1"])
    monkeypatch.setattr(evaluate_cli, "YOLO", lambda weights: FakeModel())
    with pytest.raises(SystemExit):
        evaluate_cli.main()
