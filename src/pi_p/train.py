import argparse
from pathlib import Path

from ultralytics import YOLO

from pi_p.config import load


def main():
    ap = argparse.ArgumentParser(description="Antreneaza YOLO cu parametrii dintr-un fisier de configurare")
    ap.add_argument("config", nargs="?", default="configs/train.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()

    cfg = load(args.config, args.overrides)
    # Ultralytics pune un `project` relativ sub runs/detect/, deci runs/<name> ar ajunge
    # in runs/detect/runs/<name>. Cu o cale absoluta, rezultatele ajung chiar in runs/<name>.
    if "project" in cfg:
        cfg["project"] = str(Path(cfg["project"]).resolve())

    # `model` spune de la ce greutati pornim; restul parametrilor merg direct la antrenare.
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
