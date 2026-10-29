import numpy as np
from ultralytics import YOLO

from pi_p.scoremap import ScoreMap


def test_scoremap_matches_frame_and_is_probability():
    model = YOLO("yolo26n.yaml")
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device="cpu", verbose=False)
    sm = ScoreMap(model.predictor.model.model)
    frame = np.random.default_rng(0).integers(0, 255, (360, 900, 3), dtype=np.uint8)
    model.predict(frame, device="cpu", verbose=False)
    hm = sm.heatmap(360, 900)
    assert hm.shape == (360, 900)
    assert 0 <= hm.min() <= hm.max() <= 1
