import argparse
import asyncio
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("WEBSOCKETS_MAX_LINE_LENGTH", str(64 * 1024))

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response as RawResponse
from fastapi.staticfiles import StaticFiles

from pi_p.app.library import VIDEO_EXT, list_videos, resolve, thumbnail
from pi_p.app.session import Session, model_choices
from pi_p.auth import COOKIE, SESSION_SECONDS, Auth, load_env

WORK = Path(tempfile.gettempdir()) / "pi-p"
STATIC = Path("frontend/dist")
LIBRARY = Path(os.environ.get("PI_P_LIBRARY", "videos"))
RESULTS = {"detectii.mp4", "numaratoare.csv"}
MAX_UPLOAD = 2 * 1024**3
MAX_SESSIONS = 2
MAX_AGE = 24 * 3600
ID = re.compile(r"[0-9a-f]{32}")
LIVE, CONGESTION = b"\x01", b"\x02"

PUBLIC = {"/api/login", "/api/session"}

app = FastAPI(title="pi-p")
slots = asyncio.Semaphore(MAX_SESSIONS)
auth = Auth(os.environ.get("PI_P_TOTP_SECRET"))


def client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in PUBLIC and not auth.valid(request.cookies.get(COOKIE)):
        return JSONResponse({"detail": "Autentificare necesară"}, status_code=401)
    return await call_next(request)


@app.get("/api/session")
def session_state(request: Request) -> dict:
    return {"enabled": auth.enabled, "authenticated": auth.valid(request.cookies.get(COOKIE))}


@app.post("/api/login")
async def login(request: Request, response: Response) -> dict:
    if not auth.enabled:
        return {"ok": True}
    if not auth.allow_attempt(client_ip(request)):
        raise HTTPException(429, "Prea multe încercări, așteaptă un minut")
    try:
        code = str((await request.json()).get("code", "")).strip()
    except ValueError:
        code = ""
    if not auth.verify(code):
        raise HTTPException(401, "Cod invalid")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, auth.issue(), max_age=SESSION_SECONDS, httponly=True, secure=secure, samesite="strict")
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE)
    return {"ok": True}


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


@app.get("/api/library")
def library() -> dict:
    return {"videos": list_videos(LIBRARY)}


@app.get("/api/library/{name}/thumbnail")
def library_thumbnail(name: str) -> RawResponse:
    try:
        path = resolve(LIBRARY, name)
        return RawResponse(thumbnail(path, path.stat().st_mtime), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/results/{vid}/{name}")
def result(vid: str, name: str) -> FileResponse:
    path = session_dir(vid) / name
    if name not in RESULTS or not path.is_file():
        raise HTTPException(404, "Rezultat inexistent")
    return FileResponse(path, filename=name)


def parse_start(msg: dict) -> tuple[Path, Path, str, bool, int, float, bool]:
    if msg.get("action") != "start":
        raise ValueError("Primul mesaj trebuie să fie start")
    if msg.get("library"):
        video = resolve(LIBRARY, str(msg["library"]))
        out = WORK / uuid.uuid4().hex
        out.mkdir(parents=True)
    else:
        out = session_dir(str(msg.get("video", "")))
        video = next(out.glob("input.*"))
    weights = msg.get("weights")
    if weights not in model_choices():
        raise ValueError("Model necunoscut")
    slice_size = int(msg.get("slice", 0))
    if slice_size and not 160 <= slice_size <= 2048:
        raise ValueError("Fereastra trebuie între 160 și 2048 px")
    conf = min(max(float(msg.get("conf", 0.25)), 0.05), 0.95)
    return video, out, weights, bool(msg.get("track", True)), slice_size, conf, bool(msg.get("realtime", True))


@app.websocket("/api/stream")
async def stream(ws: WebSocket) -> None:
    if not auth.valid(ws.cookies.get(COOKIE)):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        video, out, weights, track, slice_size, conf, realtime = parse_start(await ws.receive_json())
    except (ValueError, HTTPException, StopIteration) as e:
        await ws.send_json({"type": "error", "message": getattr(e, "detail", str(e))})
        await ws.close()
        return
    if slots.locked():
        await ws.send_json({"type": "error", "message": "Serverul procesează deja numărul maxim de video-uri"})
        await ws.close()
        return
    async with slots:
        await run_session(ws, video, out, weights, track, slice_size, conf, realtime)


async def run_session(ws: WebSocket, video: Path, out_dir: Path, weights: str, track: bool, slice_size: int, conf: float, realtime: bool) -> None:
    try:
        session = await asyncio.to_thread(Session, video, weights, track, slice_size, out_dir)
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return
    session.params.conf = conf
    stop = asyncio.Event()
    playing = asyncio.Event()
    playing.set()

    async def listen() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("action") == "stop":
                    stop.set()
                    playing.set()
                elif msg.get("action") == "pause":
                    playing.clear()
                elif msg.get("action") == "resume":
                    playing.set()
                elif msg.get("action") == "params":
                    session.params.conf = min(max(float(msg.get("conf", session.params.conf)), 0.05), 0.95)
                    if msg.get("view") in ("detections", "scores"):
                        session.params.view = msg["view"]
                    if not playing.is_set() and (live := await asyncio.to_thread(session.redraw)):
                        await ws.send_bytes(LIVE + live)
        except (WebSocketDisconnect, RuntimeError, ValueError):
            stop.set()
            playing.set()

    listener = asyncio.create_task(listen())
    connected = True
    try:
        await ws.send_json(session.meta(weights))
        t0, i = time.perf_counter(), 0
        while not stop.is_set():
            if not playing.is_set():
                paused_at = time.perf_counter()
                await playing.wait()
                t0 += time.perf_counter() - paused_at
                continue
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
        vid = out_dir.name
        try:
            await ws.send_json({"type": "done", "files": {f: f"/api/results/{vid}/{f}" for f in files}})
            await ws.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


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
    global auth
    secret = os.environ.get("PI_P_TOTP_SECRET") or load_env().get("PI_P_TOTP_SECRET")
    auth = Auth(secret)
    if not auth.enabled:
        print("Atenție: PI_P_TOTP_SECRET nu e setat, aplicația rulează fără autentificare")
    cleanup()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
