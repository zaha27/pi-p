import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pi_p_server import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORK", tmp_path / "work")
    monkeypatch.setattr(server, "LIBRARY", tmp_path / "videos")
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
    monkeypatch.setattr("pi_p_server.session.device", lambda: "cpu")
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


def test_pause_holds_frames_until_resume(client, video, monkeypatch):
    monkeypatch.setattr("pi_p_server.session.device", lambda: "cpu")
    vid = client.post("/api/videos", files={"file": ("clip.mp4", video.read_bytes())}).json()["id"]
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "video": vid, "weights": "yolo26n.yaml", "realtime": True})
        assert ws.receive_json()["type"] == "meta"
        ws.send_json({"action": "pause"})
        time.sleep(0.5)
        ws.send_json({"action": "resume"})
        frames = []
        while True:
            msg = ws.receive()
            data = json.loads(msg["text"]) if msg.get("text") else {}
            if data.get("type") == "metrics":
                frames.append(data)
            if data.get("type") == "done":
                break
    assert [m["frame"] for m in frames] == list(range(1, 7))
    assert frames[-1]["time"] == 0.6


def test_library_lists_and_streams_server_side_videos(client, video, tmp_path, monkeypatch):
    monkeypatch.setattr("pi_p_server.session.device", lambda: "cpu")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "drona.mp4").write_bytes(video.read_bytes())
    (tmp_path / "videos" / "notite.txt").write_text("x")
    videos = client.get("/api/library").json()["videos"]
    assert [v["name"] for v in videos] == ["drona.mp4"]
    assert videos[0]["width"] == 320 and videos[0]["duration"] == 0.6
    assert client.get("/api/library/drona.mp4/thumbnail").headers["content-type"] == "image/jpeg"
    assert client.get("/api/library/..%2Fclip.mp4/thumbnail").status_code == 404
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "library": "../clip.mp4", "weights": "yolo26n.yaml"})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"action": "start", "library": "drona.mp4", "weights": "yolo26n.yaml", "realtime": False})
        while (data := ws.receive()).get("bytes") or json.loads(data["text"])["type"] not in ("done", "error"):
            pass
        done = json.loads(data["text"])
    assert done["type"] == "done"
    assert client.get(done["files"]["numaratoare.csv"]).status_code == 200
    assert (tmp_path / "videos" / "drona.mp4").is_file()
