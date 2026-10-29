from functools import lru_cache
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
THUMB_WIDTH = 320


@lru_cache(maxsize=512)
def probe(path: Path, mtime: float) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 2),
            "duration": round(frames / fps, 1),
        }
    finally:
        cap.release()


def list_videos(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    videos = []
    for p in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            st = p.stat()
            videos.append({"name": p.name, "size": st.st_size, "modified": st.st_mtime, **probe(p, st.st_mtime)})
    return videos


def resolve(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).name != name or path.suffix.lower() not in VIDEO_EXT or not path.is_file():
        raise ValueError("Video inexistent în bibliotecă")
    return path


@lru_cache(maxsize=256)
def thumbnail(path: Path, mtime: float) -> bytes:
    cap = cv2.VideoCapture(str(path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames // 10, 250))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("Nu pot citi video-ul")
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (THUMB_WIDTH, round(h * THUMB_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
