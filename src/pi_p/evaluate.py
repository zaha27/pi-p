import argparse
from pathlib import Path

from ultralytics import YOLO

from pi_p import adaptive_eval, coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="ex. yolo26n.pt (baseline COCO, cu --coco) sau best.pt")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--coco", action="store_true", help="model cu clase COCO: mapează etichetele VisDrone pe COCO")
    ap.add_argument("--mode", choices=["standard", "plain", "adaptive"], default="standard", help="standard = model.val; plain/adaptive = evaluator propriu (imgsz din config)")
    ap.add_argument("--adaptive-config", default="configs/adaptive.yaml")
    ap.add_argument("--limit", type=int, default=None, help="doar primele N imagini (plain/adaptive)")
    ap.add_argument("--out", type=Path, default=None, help="fișier JSON cu rezultatele (plain/adaptive)")
    ap.add_argument("overrides", nargs="*", help="key=value pentru config (plain/adaptive); se scriu imediat după weights")
    args = ap.parse_args()
    model = YOLO(args.weights)
    data, classes = args.data, None
    if args.coco:
        if len(model.names) != 80:
            raise SystemExit(f"--coco cere un model cu cele 80 de clase COCO, {args.weights} are {len(model.names)}")
        data = coco.build(Path(args.data).parent, "val", model.names)
        classes = sorted(set(coco.TO_COCO.values()))
    if args.mode != "standard" or args.limit or args.out or args.overrides:
        adaptive_eval.run(model, args.mode, data, args.adaptive_config, tuple(args.overrides), args.limit, args.device, args.out, classes)
        return
    m = model.val(data=str(data), imgsz=args.imgsz, device=args.device, classes=classes)
    print(f"mAP50-95: {m.box.map:.3f}  mAP50: {m.box.map50:.3f}")
    for c in m.box.ap_class_index:
        print(f"  {m.names[c]:>16}: {m.box.maps[c]:.3f}")


if __name__ == "__main__":
    main()
