import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pi_p.app import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORK", tmp_path / "work")
    monkeypatch.setattr(server, "model_choices", lambda: ["yolo26n.yaml"])
    return TestClient(server.app)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 192))
    rng = np.random.default_rng(0)
    for _ in range(6):
        w.write(rng.integers(0, 255, (192, 320, 3), dtype=np.uint8))
    w.release()
    return path


def test_upload_rejects_unknown_format(client):
    r = client.post("/api/videos", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_stream_rejects_unknown_model(client, video):
    vid = client.post("/api/videos", files={"file": ("clip.mp4", video.read_bytes())}).json()["id"]
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "video": vid, "weights": "/etc/passwd"})
        assert ws.receive_json() == {"type": "error", "message": "Model necunoscut"}


def test_stream_end_to_end(client, video, monkeypatch):
    monkeypatch.setattr("pi_p.app.session.device", lambda: "cpu")
    vid = client.post("/api/videos", files={"file": ("clip.mp4", video.read_bytes())}).json()["id"]
    kinds, metrics = [], []
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "video": vid, "weights": "yolo26n.yaml", "realtime": False})
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                kinds.append(msg["bytes"][0])
                continue
            data = json.loads(msg["text"])
            kinds.append(data["type"])
            if data["type"] == "metrics":
                metrics.append(data)
            if data["type"] in ("done", "error"):
                break
    assert kinds[0] == "meta"
    assert kinds[-1] == "done", kinds
    assert kinds.count(1) == 6 and kinds.count(2) >= 1
    assert [m["frame"] for m in metrics] == list(range(1, 7))
    files = data["files"]
    assert client.get(files["numaratoare.csv"]).text.startswith("frame,")
    assert client.get(f"/api/results/{vid}/../../etc").status_code == 404


def test_pause_stops_frames_until_resume(client, video, monkeypatch):
    monkeypatch.setattr("pi_p.app.session.device", lambda: "cpu")
    vid = client.post("/api/videos", files={"file": ("clip.mp4", video.read_bytes())}).json()["id"]
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "video": vid, "weights": "yolo26n.yaml", "realtime": True})
        assert ws.receive_json()["type"] == "meta"
        ws.send_json({"action": "pause"})
        frames = []
        while True:
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"])["type"] == "metrics":
                frames.append(json.loads(msg["text"])["frame"])
                if len(frames) == 1:
                    ws.send_json({"action": "resume"})
            if msg.get("text") and json.loads(msg["text"])["type"] == "done":
                break
    assert frames == list(range(1, 7))
