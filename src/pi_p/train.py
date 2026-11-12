"""Antrenarea modelului: fine-tuning YOLO26n pe VisDrone.

`pi-train` citește un fișier din `configs/` și pornește antrenarea Ultralytics
cu parametrii de acolo. Modelul pleacă de la greutățile preantrenate pe COCO
(`yolo26n.pt`) și e adaptat la cele 10 clase VisDrone.

    pi-train configs/smoke.yaml         test rapid: 1 epocă pe 5% din date
    pi-train configs/train.yaml         baseline: imagini întregi
    pi-train configs/train_tiled.yaml   pe felii (rulează întâi pi-tile)

Rezultatele ajung în `runs/<name>/`; cel mai bun model este
`weights/best.pt`. Detalii în docs/pipeline/05-antrenarea.md.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from pi_p.config import load


def main():
    ap = argparse.ArgumentParser(description="Antrenează YOLO cu parametrii dintr-un fișier de configurare")
    ap.add_argument("config", nargs="?", default="configs/train.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()

    cfg = load(args.config, args.overrides)
    # Ultralytics pune un `project` relativ sub runs/detect/, deci runs/<name> ar ajunge
    # în runs/detect/runs/<name>. Cu o cale absolută, rezultatele ajung chiar în runs/<name>.
    if "project" in cfg:
        cfg["project"] = str(Path(cfg["project"]).resolve())

    # `model` spune de la ce greutăți pornim; restul parametrilor merg direct la antrenare.
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
