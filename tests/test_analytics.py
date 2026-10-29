import json

import numpy as np

from pi_p.analytics import JAM_LEVEL, Analytics


def frame(boxes, cls, ids=None):
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(cls, dtype=int), None if ids is None else np.array(ids)


def test_flow_unique_tracks_and_class_shares():
    a = Analytics({0: "pedestrian", 3: "car"}, 1000, 500, 10)
    for t in range(5):
        a.update(*frame([[100 + 10 * t, 100, 120 + 10 * t, 120], [500, 300, 540, 330]], [3, 0], [1, 2]), 12.0)
    s = a.summary()
    assert s["unique"] == 2
    assert s["in_frame"] == 2
    assert [c["id"] for c in s["classes"]] == [0, 3]
    assert [c["share"] for c in s["classes"]] == [0.5, 0.5]
    assert np.isclose(s["mean_flow"], 5.0)
    json.dumps(s)
    moving = next(f for f in s["flow"] if f[4] == 3)
    assert np.isclose(moving[2] * 1000, 10.0)


def test_parked_object_saturates_and_moving_does_not():
    a = Analytics({0: "car"}, 960, 540, 5)
    for t in range(400):
        x = (t * 37) % 900
        a.update(*frame([[100, 100, 200, 160], [x, 400, x + 40, 430]], [0, 0]), 1.0)
    occ = a.occupancy()
    s = a.summary()
    assert occ[int(130 * a.scale), int(150 * a.scale)] >= JAM_LEVEL
    assert occ[int(415 * a.scale), int(450 * a.scale)] < JAM_LEVEL
    assert 0 < s["saturation"] < 0.1
    px, py, _ = s["peak"]
    assert 80 / 960 < px < 220 / 960 and 80 / 540 < py < 180 / 540


def test_without_tracking_there_is_no_flow():
    a = Analytics({0: "car"}, 100, 100, 25)
    a.update(*frame([[10, 10, 20, 20]], [0]), 1.0)
    s = a.summary()
    assert s["unique"] == 0 and s["flow"] == [] and s["mean_flow"] == 0
