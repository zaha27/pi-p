"""Evaluarea: compară detecțiile cu etichetele făcute de oameni și calculează mAP.

Toate variantele (imagine întreagă, grilă, ferestre adaptive) sunt evaluate
la fel: pe imaginile ÎNTREGI, după fuziunea detecțiilor, cu același prag de
încredere. Numai așa rezultatele se pot compara între ele.

Pașii, pentru fiecare imagine:

  1. rulăm detecția în modul cerut (vezi predict.detect);
  2. potrivim detecțiile cu obiectele adnotate (`match`);
  3. ținem minte, pentru fiecare detecție, dacă e corectă, scorul și clasa.

La final, din toate imaginile, calculăm precizia medie pe clasă (AP) și media
lor (mAP), pe tot setul și separat pe grupele de iluminare.

    pi-eval runs/.../best.pt                       imagine întreagă
    pi-eval runs/.../best.pt --mode grila          pe grila de ferestre
    pi-eval yolo26n.pt --coco                      model COCO, neantrenat pe VisDrone

Detalii în docs/pipeline/08-evaluarea.md.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywhn2xyxy

from pi_p import coco
from pi_p.data import NAMES, images, label_path, read_labels
from pi_p.predict import MAX_DET, MODES, detect
from pi_p.stats import GROUPS, read_groups

# Pragurile IoU la care verificăm o detecție: 0.50, 0.55, ..., 0.95.
# mAP50 folosește doar primul prag; mAP50-95 este media pe toate cele zece.
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# Primele imagini sunt mai lente (se încarcă modelul pe placa video), deci nu le cronometrăm.
WARMUP_IMAGES = 5


def match(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Decide care detecții sunt corecte, la fiecare prag IoU.

    `iou` are o linie pentru fiecare obiect adnotat și o coloană pentru
    fiecare detecție. O detecție e corectă dacă are aceeași clasă cu un obiect
    adnotat și IoU peste prag. Fiecare obiect adnotat poate fi „folosit” de o
    singură detecție, iar fiecare detecție de un singur obiect; perechile cu
    IoU mai mare au prioritate. Este aceeași regulă ca în validarea Ultralytics.

    Întoarce un tabel True/False: o linie pe detecție, o coloană pe prag.
    """
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    # Perechile cu clase diferite nu se pot potrivi, deci le punem IoU 0.
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])

    for t, threshold in enumerate(IOU_THRESHOLDS):
        pairs = np.array(np.nonzero(iou >= threshold)).T  # linii (obiect, detecție)
        if len(pairs) == 0:
            continue
        if len(pairs) > 1:
            # Sortăm perechile descrescător după IoU, apoi păstrăm prima pereche
            # pentru fiecare detecție și prima pentru fiecare obiect.
            pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        correct[pairs[:, 1], t] = True
    return correct


def evaluate_image(det: torch.Tensor, labels: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
    """Compară detecțiile unei imagini cu etichetele ei."""
    gt_cls = labels[:, 0].astype(int)
    gt_boxes = torch.from_numpy(xywhn2xyxy(labels[:, 1:], w=w, h=h)).float()

    # Păstrăm cel mult MAX_DET detecții, pe cele cu scorul cel mai mare.
    det = det[det[:, 4].argsort(descending=True)][:MAX_DET].float()
    pred_cls = det[:, 5].numpy().astype(int)

    if len(gt_boxes) and len(det):
        correct = match(pred_cls, gt_cls, box_iou(gt_boxes, det[:, :4]).numpy())
    else:
        correct = np.zeros((len(det), len(IOU_THRESHOLDS)), dtype=bool)
    return {"correct": correct, "conf": det[:, 4].numpy(), "pred_cls": pred_cls, "gt_cls": gt_cls}


def summarize(per_image: list[dict[str, np.ndarray]]) -> dict | None:
    """Calculează precizia, recall-ul și mAP din rezultatele mai multor imagini.

    AP pe o clasă este aria de sub curba precizie-recall, obținută coborând
    treptat pragul de încredere; calculul îl face `ap_per_class` din Ultralytics.
    """
    if not per_image:
        return None
    correct = np.concatenate([r["correct"] for r in per_image])
    conf = np.concatenate([r["conf"] for r in per_image])
    pred_cls = np.concatenate([r["pred_cls"] for r in per_image])
    gt_cls = np.concatenate([r["gt_cls"] for r in per_image])
    if len(gt_cls) == 0 or len(conf) == 0:
        return None

    result = ap_per_class(correct, conf, pred_cls, gt_cls, names=dict(enumerate(NAMES)))
    precision, recall, ap, classes = result[2], result[3], result[5], result[6]
    return {
        "imagini": len(per_image),
        "P": float(precision.mean()),
        "R": float(recall.mean()),
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "pe_clase": {NAMES[c]: {"mAP50": float(ap[i, 0]), "mAP50-95": float(ap[i].mean())} for i, c in enumerate(classes)},
    }


def evaluate(model: YOLO, root: Path, split: str, mode: str, size: int, overlap: float, conf: float, iou: float,
             device, groups: dict[str, str] | None = None) -> dict:
    """Evaluează un model pe un subset și întoarce toate rezultatele ca dicționar."""
    paths = images(root, split)
    per_image, times = [], []

    for k, path in enumerate(paths):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]

        # Așteptăm placa video înainte și după, ca timpul măsurat să fie cel real.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        det = detect(model, img, mode, size, overlap, conf, iou, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if k >= WARMUP_IMAGES:
            times.append(time.perf_counter() - start)

        per_image.append(evaluate_image(det, read_labels(label_path(path)), w, h))
        if k % 100 == 0:
            print(f"{k}/{len(paths)} imagini", flush=True)

    results = {"mod": mode, "subset": split, "conf": conf, **summarize(per_image)}
    results["ms_pe_imagine"] = 1000 * float(np.mean(times)) if times else None

    # Aceleași calcule, separat pe grupele de iluminare (dacă există tabelul de la pi-stats).
    if groups:
        results["pe_grupe"] = {}
        for group in GROUPS:
            selected = [r for path, r in zip(paths, per_image) if groups.get(path.stem) == group]
            summary = summarize(selected)
            if summary:
                results["pe_grupe"][group] = {k: summary[k] for k in ("imagini", "mAP50", "mAP50-95")}
    return results


def print_results(results: dict) -> None:
    """Afișează rezultatele ca text, pentru terminal."""
    print(f"\nmod: {results['mod']}   subset: {results['subset']}   imagini: {results['imagini']}")
    print(f"mAP50: {results['mAP50']:.3f}   mAP50-95: {results['mAP50-95']:.3f}   "
          f"P: {results['P']:.3f}   R: {results['R']:.3f}")
    if results["ms_pe_imagine"]:
        print(f"timp: {results['ms_pe_imagine']:.1f} ms/imagine")
    for name, m in results["pe_clase"].items():
        print(f"  {name:>16}: mAP50 {m['mAP50']:.3f}   mAP50-95 {m['mAP50-95']:.3f}")
    for group, m in results.get("pe_grupe", {}).items():
        print(f"  grupa {group:>14} ({m['imagini']} imagini): mAP50 {m['mAP50']:.3f}   mAP50-95 {m['mAP50-95']:.3f}")


def evaluate_coco(weights: str, data: str, imgsz: int, device) -> None:
    """Evaluează un model cu clase COCO (neantrenat pe VisDrone), prin maparea claselor."""
    model = YOLO(weights)
    if len(model.names) != 80:
        raise SystemExit(f"--coco cere un model cu cele 80 de clase COCO, {weights} are {len(model.names)}")
    mapped_data = coco.build(Path(data).parent, "val", model.names)
    classes = sorted(set(coco.TO_COCO.values()))
    m = model.val(data=str(mapped_data), imgsz=imgsz, device=device, classes=classes)
    print(f"mAP50-95: {m.box.map:.3f}  mAP50: {m.box.map50:.3f}")
    for c in m.box.ap_class_index:
        print(f"  {m.names[c]:>16}: {m.box.maps[c]:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluează un model pe imaginile întregi ale unui subset")
    ap.add_argument("weights", help="ex. yolo26n.pt (baseline COCO, cu --coco) sau best.pt")
    ap.add_argument("--data", default="datasets/VisDrone/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--mode", choices=MODES, default="intreg")
    ap.add_argument("--imgsz", type=int, default=640, help="latura imaginii micșorate și a ferestrelor")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.01, help="prag de încredere; același, și mic, pentru toate variantele")
    ap.add_argument("--iou", type=float, default=0.5, help="prag IOS la fuziunea detecțiilor din ferestre")
    ap.add_argument("--device", default=None)
    ap.add_argument("--groups", type=Path, default=None, help="tabel scris de pi-stats; implicit datasets/VisDrone/stats/<subset>.csv")
    ap.add_argument("--out", type=Path, default=None, help="fișier JSON în care se salvează rezultatele")
    ap.add_argument("--coco", action="store_true", help="model cu clase COCO: mapează etichetele VisDrone pe COCO")
    args = ap.parse_args()

    if args.coco:
        evaluate_coco(args.weights, args.data, args.imgsz, args.device)
        return

    # Folderul setului de date este scris în data.yaml, la cheia `path`.
    root = Path(yaml.safe_load(Path(args.data).read_text())["path"])
    groups_file = args.groups or Path("datasets/VisDrone/stats") / f"{args.split}.csv"
    groups = read_groups(groups_file) if groups_file.is_file() else None

    model = YOLO(args.weights)
    results = evaluate(model, root, args.split, args.mode, args.imgsz, args.overlap, args.conf, args.iou, args.device, groups)
    results["model"] = args.weights
    results["date"] = args.data
    print_results(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Rezultate salvate în {args.out}")


if __name__ == "__main__":
    main()
