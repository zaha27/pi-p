import argparse

from ultralytics import YOLO

from pi_p.config import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/train.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()
    cfg = load(args.config, args.overrides)
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
