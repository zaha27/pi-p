import argparse
import asyncio
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pi_p.app.session import Session, model_choices

WORK = Path(tempfile.gettempdir()) / "pi-p"
STATIC = Path("frontend/dist")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
RESULTS = {"detectii.mp4", "numaratoare.csv"}
MAX_UPLOAD = 2 * 1024**3
MAX_SESSIONS = 2
MAX_AGE = 24 * 3600
ID = re.compile(r"[0-9a-f]{32}")
LIVE, CONGESTION = b"\x01", b"\x02"

app = FastAPI(title="pi-p")
slots = asyncio.Semaphore(MAX_SESSIONS)


def session_dir(vid: str) -> Path:
    if not ID.fullmatch(vid) or not (WORK / vid).is_dir():
        raise HTTPException(404, "Video inexistent")
    return WORK / vid


@app.get("/api/models")
def models() -> dict:
    return {"models": model_choices()}


@app.post("/api/videos")
async def upload(file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"Format nesuportat; acceptate: {', '.join(sorted(VIDEO_EXT))}")
    vid = uuid.uuid4().hex
    d = WORK / vid
    d.mkdir(parents=True)
    size = 0
    with open(d / f"input{ext}", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(d)
                raise HTTPException(413, "Fișier prea mare (max 2 GB)")
            f.write(chunk)
    return {"id": vid}


@app.get("/api/results/{vid}/{name}")
def result(vid: str, name: str) -> FileResponse:
    path = session_dir(vid) / name
    if name not in RESULTS or not path.is_file():
        raise HTTPException(404, "Rezultat inexistent")
    return FileResponse(path, filename=name)


def parse_start(msg: dict) -> tuple[Path, str, bool, int, float, bool]:
    if msg.get("action") != "start":
        raise ValueError("Primul mesaj trebuie să fie start")
    d = session_dir(str(msg.get("video", "")))
    inputs = list(d.glob("input.*"))
    weights = msg.get("weights")
    if weights not in model_choices():
        raise ValueError("Model necunoscut")
    slice_size = int(msg.get("slice", 0))
    if slice_size and not 160 <= slice_size <= 2048:
        raise ValueError("Fereastra trebuie între 160 și 2048 px")
    conf = min(max(float(msg.get("conf", 0.25)), 0.05), 0.95)
    return inputs[0], weights, bool(msg.get("track", True)), slice_size, conf, bool(msg.get("realtime", True))


@app.websocket("/api/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        video, weights, track, slice_size, conf, realtime = parse_start(await ws.receive_json())
    except (ValueError, HTTPException, IndexError) as e:
        await ws.send_json({"type": "error", "message": getattr(e, "detail", str(e))})
        await ws.close()
        return
    if slots.locked():
        await ws.send_json({"type": "error", "message": "Serverul procesează deja numărul maxim de video-uri"})
        await ws.close()
        return
    async with slots:
        await run_session(ws, video, weights, track, slice_size, conf, realtime)


async def run_session(ws: WebSocket, video: Path, weights: str, track: bool, slice_size: int, conf: float, realtime: bool) -> None:
    try:
        session = await asyncio.to_thread(Session, video, weights, track, slice_size, video.parent)
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return
    session.params.conf = conf
    stop = asyncio.Event()

    async def listen() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("action") == "stop":
                    stop.set()
                elif msg.get("action") == "params":
                    session.params.conf = min(max(float(msg.get("conf", session.params.conf)), 0.05), 0.95)
                    if msg.get("view") in ("detections", "scores"):
                        session.params.view = msg["view"]
        except (WebSocketDisconnect, RuntimeError, ValueError):
            stop.set()

    listener = asyncio.create_task(listen())
    connected = True
    try:
        await ws.send_json(session.meta(weights))
        t0, i = time.perf_counter(), 0
        while not stop.is_set():
            out = await asyncio.to_thread(session.step)
            if out is None:
                break
            live, congestion, summary = out
            i += 1
            if realtime and (delay := t0 + i / session.fps - time.perf_counter()) > 0:
                await asyncio.sleep(delay)
            await ws.send_bytes(LIVE + live)
            if congestion:
                await ws.send_bytes(CONGESTION + congestion)
            await ws.send_json({"type": "metrics", "time": i / session.fps, **summary})
    except (WebSocketDisconnect, RuntimeError):
        connected = False
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        connected = False
    finally:
        listener.cancel()
        files = await asyncio.to_thread(session.close)
    if connected:
        vid = video.parent.name
        await ws.send_json({"type": "done", "files": {f: f"/api/results/{vid}/{f}" for f in files}})
        await ws.close()


def cleanup() -> None:
    if not WORK.is_dir():
        return
    for d in WORK.iterdir():
        if d.is_dir() and time.time() - d.stat().st_mtime > MAX_AGE:
            shutil.rmtree(d, ignore_errors=True)


if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    cleanup()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
