"""Verificarea unui set de date YOLO înainte de antrenare (`pi-check`)."""

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pi_p.data import NAMES, ROOT, images, label_path

# Toleranță la comparația cu marginile imaginii, pentru erorile de rotunjire.
EPS = 1e-3


def check_label_file(path: Path, nc: int) -> list[tuple[str, int]]:
    """Verifică un fișier de etichete; întoarce perechi (tipul problemei, numărul liniei)."""
    issues = []
    seen = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(("format", line_no))
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            issues.append(("format", line_no))
            continue

        if not 0 <= cls < nc:
            issues.append(("clasă invalidă", line_no))

        if w <= 0 or h <= 0:
            issues.append(("arie zero", line_no))
        else:
            left, top = x - w / 2, y - h / 2
            right, bottom = x + w / 2, y + h / 2
            if min(left, top) < -EPS or max(right, bottom) > 1 + EPS:
                issues.append(("în afara imaginii", line_no))

        if line in seen:
            issues.append(("duplicat", line_no))
        seen.add(line)
    return issues


def check_split(root: Path, split: str, nc: int) -> dict[str, list[str]]:
    """Verifică un subset întreg; întoarce, pe tipuri de probleme, unde au apărut."""
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
        for kind, line_no in check_label_file(lbl, nc):
            found[kind].append(f"{lbl.name}:{line_no}")

    # Și invers: fișiere de etichete pentru care nu există imagine.
    stems = {p.stem for p in imgs}
    for lbl in (root / "labels" / split).glob("*.txt"):
        if lbl.stem not in stems:
            found["etichetă fără imagine"].append(lbl.name)
    return found


def main():
    ap = argparse.ArgumentParser(description="Verifică un set de date în format YOLO")
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
