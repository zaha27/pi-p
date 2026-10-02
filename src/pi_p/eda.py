import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pi_p.data import NAMES, ROOT, image_size, images, label_path, read_labels

COLORS = (np.array(plt.get_cmap("tab10").colors) * 255).astype(int)


def collect(imgs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    objs, per_img = [], []
    for img in imgs:
        w, h = image_size(img)
        lbl = read_labels(label_path(img))
        per_img.append(len(lbl))
        objs.append(np.column_stack([lbl[:, :3], lbl[:, 3] * w, lbl[:, 4] * h, np.full(len(lbl), max(w, h))]))
    return np.concatenate(objs), np.array(per_img)


def draw(img_path: Path) -> np.ndarray:
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    for c, x, y, bw, bh in read_labels(label_path(img_path)):
        color = tuple(int(v) for v in COLORS[int(c)][::-1])
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), color, 2)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def summarize(objs: np.ndarray, per_img: np.ndarray, sizes: list[int]) -> dict:
    cls, wpx, hpx, longest = objs[:, 0].astype(int), objs[:, 3], objs[:, 4], objs[:, 5]
    area = wpx * hpx
    side = np.sqrt(area)
    s = {
        "images": len(per_img),
        "objects": len(objs),
        "objects_per_image": {"mean": float(per_img.mean()), "median": float(np.median(per_img)), "max": int(per_img.max())},
        "classes": {NAMES[c]: int((cls == c).sum()) for c in range(len(NAMES))},
        "coco_buckets_native": {
            "small": float((area < 32**2).mean()),
            "medium": float(((area >= 32**2) & (area < 96**2)).mean()),
            "large": float((area >= 96**2).mean()),
        },
        "side_px": {"native (tiling)": float(np.median(side))},
        "under_16px": {"native (tiling)": float((side < 16).mean())},
    }
    for sz in sizes:
        scaled = side * sz / longest
        s["side_px"][f"imgsz {sz}"] = float(np.median(scaled))
        s["under_16px"][f"imgsz {sz}"] = float((scaled < 16).mean())
    return s


def plot(objs: np.ndarray, per_img: np.ndarray, sizes: list[int], imgs: list[Path], out: Path) -> None:
    cls, side, longest = objs[:, 0].astype(int), np.sqrt(objs[:, 3] * objs[:, 4]), objs[:, 5]
    counts = np.bincount(cls, minlength=len(NAMES))
    order = np.argsort(counts)
    bins = np.logspace(0, 3, 60)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].barh(np.array(NAMES)[order], counts[order], color=COLORS[order] / 255)
    ax[0].set_title("Instanțe pe clasă")
    ax[1].hist(side, bins=bins, histtype="step", lw=2, label="nativ (tiling)")
    for sz in sizes:
        ax[1].hist(side * sz / longest, bins=bins, histtype="step", lw=2, label=f"imgsz {sz}")
    ax[1].set_xscale("log")
    ax[1].axvline(16, color="r", ls="--", lw=1)
    ax[1].set_title("Latura obiectului (px) văzută de model")
    ax[1].legend()
    ax[2].hist(per_img, bins=50)
    ax[2].set_title("Obiecte pe imagine")
    fig.tight_layout()
    fig.savefig(out / "stats.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].boxplot([side[cls == c] for c in range(len(NAMES))], tick_labels=NAMES, showfliers=False)
    ax[0].set_yscale("log")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].set_title("Latura nativă (px) pe clasă")
    ax[1].hist2d(objs[:, 1], objs[:, 2], bins=64, range=[[0, 1], [0, 1]], cmap="magma")
    ax[1].invert_yaxis()
    ax[1].set_aspect("equal")
    ax[1].set_title("Unde apar obiectele în cadru")
    fig.tight_layout()
    fig.savefig(out / "sizes_heatmap.png", dpi=110)

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for a, img in zip(ax.flat, random.Random(0).sample(imgs, min(4, len(imgs)))):
        a.imshow(draw(img))
        a.set_title(img.name)
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i] / 255) for i in range(len(NAMES))]
    fig.legend(handles, NAMES, loc="lower center", ncol=len(NAMES))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "samples.png", dpi=110)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1024])
    ap.add_argument("--out", type=Path, default=Path("eda_out"))
    args = ap.parse_args()
    imgs = images(args.root, args.split)
    objs, per_img = collect(imgs)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(objs, per_img, args.imgsz)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot(objs, per_img, args.imgsz, imgs, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Grafice în {args.out}/")


if __name__ == "__main__":
    main()
