import argparse
from pathlib import Path

from ultralytics import YOLO

from pi_p import coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="ex. yolo26n.pt (baseline COCO, cu --coco) sau best.pt")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--coco", action="store_true", help="model cu clase COCO: mapează etichetele VisDrone pe COCO")
    args = ap.parse_args()
    model = YOLO(args.weights)
    data, classes = args.data, None
    if args.coco:
        if len(model.names) != 80:
            raise SystemExit(f"--coco cere un model cu cele 80 de clase COCO, {args.weights} are {len(model.names)}")
        data = coco.build(Path(args.data).parent, "val", model.names)
        classes = sorted(set(coco.TO_COCO.values()))
    m = model.val(data=str(data), imgsz=args.imgsz, device=args.device, classes=classes)
    print(f"mAP50-95: {m.box.map:.3f}  mAP50: {m.box.map50:.3f}")
    for c in m.box.ap_class_index:
        print(f"  {m.names[c]:>16}: {m.box.maps[c]:.3f}")


if __name__ == "__main__":
    main()
