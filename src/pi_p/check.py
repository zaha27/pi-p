import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pi_p.data import NAMES, ROOT, images, label_path

EPS = 1e-3


def check_label_file(path: Path, nc: int) -> list[tuple[str, int]]:
    issues, seen = [], set()
    for i, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(("format", i))
            continue
        try:
            c, (x, y, w, h) = int(parts[0]), map(float, parts[1:])
        except ValueError:
            issues.append(("format", i))
            continue
        if not 0 <= c < nc:
            issues.append(("clasă invalidă", i))
        if w <= 0 or h <= 0:
            issues.append(("arie zero", i))
        elif min(x - w / 2, y - h / 2) < -EPS or max(x + w / 2, y + h / 2) > 1 + EPS:
            issues.append(("în afara imaginii", i))
        if line in seen:
            issues.append(("duplicat", i))
        seen.add(line)
    return issues


def check_split(root: Path, split: str, nc: int) -> dict[str, list[str]]:
    found = defaultdict(list)
    imgs = images(root, split)
    for img in imgs:
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:
            found["imagine coruptă"].append(img.name)
        lbl = label_path(img)
        if not lbl.is_file():
            found["fără etichetă"].append(img.name)
            continue
        for kind, line in check_label_file(lbl, nc):
            found[kind].append(f"{lbl.name}:{line}")
    stems = {p.stem for p in imgs}
    for lbl in (root / "labels" / split).glob("*.txt"):
        if lbl.stem not in stems:
            found["etichetă fără imagine"].append(lbl.name)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()
    total = 0
    for split in args.splits:
        found = check_split(args.root, split, len(NAMES))
        print(f"{split}: {'OK' if not found else ''}")
        for kind, where in found.items():
            print(f"  {kind}: {len(where)} (ex. {', '.join(where[:3])})")
            total += len(where)
    if total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
