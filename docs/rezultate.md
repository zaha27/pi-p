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

## Reglarea ferestrelor adaptive

Ferestrele adaptive pot fi reglate spre precizie sau spre cost. Toate rândurile sunt pe tot `val`, cu același model:

| Setare | mAP50 | mAP50-95 | ms / imagine | Ferestre / imagine | Pixeli procesați |
|---|---|---|---|---|---|
| grilă fixă (`slice`), referință | 0.305 | 0.174 | 60 | 6 (la 1360×765) | ≈3.4x |
| implicit (`dilate_kernel=15`, `min_size=640`) | **0.327** | **0.185** | 154 | 10.5 | 5.66x |
| `dilate_kernel=31` | 0.305 | 0.171 | 84 | 5.5 | 3.45x |
| `dilate_kernel=31 min_size=480` | 0.301 | 0.168 | 74 | 5.5 | **2.56x** |

- Cu setările implicite, metoda adaptivă bate grila cu 0.02 mAP50, dar procesează mai mulți pixeli.
- Cu ferestre mai puține și mai mici, ajunge la aceeași precizie ca grila cu cu 25% mai puțini pixeli. Timpul rămâne totuși mai mare, pentru că se adaugă detecția preliminară, iar ferestrele de mărimi diferite se procesează mai puțin eficient decât cele egale ale grilei.
- Pe primele 100 de imagini din `val`, `min_size=480` părea să dea mAP50 0.350. Pe tot setul a scăzut la 0.301. Primele 100 de imagini vin din câteva secvențe video, deci nu sunt reprezentative, iar reglajele trebuie verificate pe tot `val`.

## Figuri

Pașii metodei adaptive pe o imagine de noapte din `val` (`0000117_02708_d_0000090.jpg`), cu setările implicite: 62 de casete preliminare, 15 ferestre, 150 de detecții finale.

![Pașii ferestrelor adaptive](../results/figuri/ferestre_adaptive_pasi.png)

![Fără ferestre comparat cu ferestrele adaptive](../results/figuri/ferestre_adaptive_comparatie.png)

Pixelii procesați pentru `slice` și `sahi` sunt calculați pentru imaginile de 1360×765 (6 ferestre de 640 px plus imaginea întreagă); evaluatorul îi măsoară direct doar la `adaptive`.
