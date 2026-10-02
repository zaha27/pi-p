import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="ex. yolo26n.pt (baseline COCO) sau best.pt")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    m = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50-95: {m.box.map:.3f}  mAP50: {m.box.map50:.3f}")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"  {m.names[c]:>16}: {m.box.maps[c]:.3f}")


if __name__ == "__main__":
    main()
