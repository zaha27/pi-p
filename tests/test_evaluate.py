import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from pi_p import evaluate
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
# interpolarea în 101 puncte din ultralytics (compute_ap) dă 0.995, nu 1.0, chiar și pentru predicții perfecte
PERFECT_AP = 0.995
CONFIG = "mask_size: 640\ndilate_kernel: 3\nmargin: 0\nmin_size: 640\nprelim_imgsz: 640\nprelim_conf: 0.25\nwindow_imgsz: 640\nconf: 0.25\niou: 0.5\n"


class FakeModel:
    names = NAMES

    def __init__(self, junk: int = 0):
        self.junk = junk

    def predict(self, source, imgsz, conf, device, verbose):
        source = [source] if isinstance(source, np.ndarray) else source
        out = []
        for img in source:
            # decupajele ferestrelor nu au intrare în ORACLE: modelul nu vede nimic în ele
            det = ORACLE.get(img.shape[:2], torch.zeros((0, 6))).clone()
            junk = torch.tensor([[0, 0, 5, 5, 0.99, 7]], dtype=torch.float32).repeat(self.junk, 1)
            out.append(SimpleNamespace(boxes=SimpleNamespace(data=torch.cat([det, junk]))))
        return out


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


def run_cli(monkeypatch, tmp_path, *argv, model=None):
    out = tmp_path / "out" / "result.json"
    monkeypatch.setattr("sys.argv", ["pi-eval", "w.pt", *argv, "--out", str(out)])
    monkeypatch.setattr(evaluate, "YOLO", lambda weights: model or FakeModel())
    evaluate.main()
    return json.loads(out.read_text())


def test_full_mode_perfect_predictions_give_perfect_map(val, tmp_path, monkeypatch):
    result = run_cli(monkeypatch, tmp_path, "--mode", "full", "--data", str(val))
    assert result["images"] == 2
    assert result["map50_95"] == pytest.approx(PERFECT_AP)
    assert result["ap_per_class"] == pytest.approx({"class0": PERFECT_AP, "class3": PERFECT_AP})
    assert result["windows_per_image"] == 0


def test_cap_keeps_highest_scores(val, tmp_path, monkeypatch):
    # 301 casete false cu scor mai mare decât cele bune ocupă tot plafonul de 300
    result = run_cli(monkeypatch, tmp_path, "--mode", "full", "--data", str(val), model=FakeModel(junk=301))
    assert result["max_det"] == 300
    assert result["map50_95"] == pytest.approx(0.0)


def test_adaptive_mode_counts_windows_and_uses_eval_conf(val, tmp_path, monkeypatch):
    cfg = tmp_path / "adaptive.yaml"
    cfg.write_text(CONFIG)
    result = run_cli(monkeypatch, tmp_path, "--mode", "adaptive", "--data", str(val), "--adaptive-config", str(cfg))
    assert result["map50_95"] == pytest.approx(PERFECT_AP)
    assert result["windows_per_image"] == pytest.approx(1.5)
    assert result["window_area"] == pytest.approx(1.5)
    assert result["params"]["conf"] == 0.001


def test_adaptive_override_beats_conf(val, tmp_path, monkeypatch):
    cfg = tmp_path / "adaptive.yaml"
    cfg.write_text(CONFIG)
    result = run_cli(monkeypatch, tmp_path, "conf=0.05", "--mode", "adaptive", "--data", str(val), "--adaptive-config", str(cfg))
    assert result["params"]["conf"] == 0.05


def test_groups_report_map_per_group(val, tmp_path, monkeypatch):
    stats = tmp_path / "stats.csv"
    stats.write_text("split,image,brightness,contrast,group\nval,a.jpg,50,40,dark\nval,b.jpg,100,40,normal\n")
    result = run_cli(monkeypatch, tmp_path, "--mode", "full", "--data", str(val), "--groups", str(stats))
    assert {g: r["images"] for g, r in result["groups"].items()} == {"dark": 1, "normal": 1}


@pytest.mark.parametrize("argv", [["--limit", "1"], ["conf=0.1", "--mode", "full"], ["--mode", "full", "--coco"]],
                         ids=["val_with_limit", "overrides_outside_adaptive", "coco_outside_val"])
def test_invalid_combinations_are_rejected(val, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["pi-eval", "w.pt", *argv, "--data", str(val)])
    monkeypatch.setattr(evaluate, "YOLO", lambda weights: FakeModel())
    with pytest.raises(SystemExit):
        evaluate.main()
