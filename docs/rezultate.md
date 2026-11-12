# Rezultate

## Preliminar: modelul antrenat pe Mac

Model: YOLO26n, 10 epoci pe `train`, imagini întregi de 640 px (`runs/yolo26n_mac`). Evaluare pe `val` (548 de imagini), pe imaginea întreagă, cu `pi-eval`, prag `conf 0.001`. Rezultatele complete sunt în `results/mac_*.json`.

Sunt rezultate preliminare: modelul e antrenat de 10 ori mai puțin decât cel din metodologie, iar varianta „grilă fixă” folosește tot modelul pe imagini întregi, nu cel antrenat pe felii.

| Variantă | `--mode` | mAP50-95 | mAP50 | ms / imagine (Mac, MPS) | Pixeli procesați |
|---|---|---|---|---|---|
| Baseline | `full` | 0.113 | 0.202 | 10 | 1.00x |
| Grilă fixă | `slice` | 0.174 | 0.305 | 60 | ≈3.4x |
| SAHI | `sahi` | 0.166 | 0.298 | 103 | ≈3.4x |
| Ferestre adaptive | `adaptive` | **0.185** | **0.327** | 154 | 5.66x |

mAP50 pe grupe:

| Variantă | dark (13) | low_contrast (104) | normal (431) |
|---|---|---|---|
| `full` | 0.300 | 0.217 | 0.196 |
| `slice` | 0.398 | 0.299 | 0.304 |
| `sahi` | 0.380 | 0.295 | 0.296 |
| `adaptive` | 0.408 | 0.316 | 0.326 |

## Ce arată

- Felierea crește mAP50 cu aproximativ 50% față de baseline (0.202 → 0.305), confirmând că obiectele mici sunt problema principală.
- Fuziunea proprie (`slice`) iese puțin peste SAHI, cu aceleași ferestre, și e mai rapidă.
- Ferestrele adaptive au cel mai bun mAP, dar cu parametrii din `configs/adaptive.yaml` procesează **mai mulți** pixeli decât grila (în medie 10.5 ferestre de cel puțin 640 px, care se suprapun), deci nu economisesc timp. Câștigul de precizie vine probabil din ferestrele centrate pe grupurile de obiecte. Pentru economie de timp trebuie reglate `dilate_kernel` și `min_size`, ca ferestrele să fie mai puține.
- Grupa `dark` are doar 13 imagini pe val, deci valorile ei nu sunt concludente.

Pixelii procesați pentru `slice` și `sahi` sunt calculați pentru imaginile de 1360×765 (6 ferestre de 640 px plus imaginea întreagă); evaluatorul îi măsoară direct doar la `adaptive`.
