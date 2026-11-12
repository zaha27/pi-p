# Rezultate

Toate valorile sunt masurate cu `pi-eval` pe cele 548 de imagini intregi din **val**, cu prag de incredere 0,001 si grupele din `datasets/VisDrone/stats.csv`. Cum se calculeaza fiecare coloana este explicat in [pipeline/08-evaluarea.md](pipeline/08-evaluarea.md). Subsetul test-dev nu a fost rulat inca: el se ruleaza o singura data, dupa ce toate deciziile sunt luate.

## Starea experimentelor

| Etapa din [plan](benchmarkuri.md) | Stare |
|---|---|
| 1. Impartirea imaginii | **facuta** cu antrenari complete (RTX); SAHI masurat doar pe modelul de pe Mac |
| 2. CLAHE | **facuta** pe val (ambele variante reantrenate si evaluate); ramane de verificat pe test |
| 3. Ferestre adaptive | masurate pe ambele modele; reglarea parametrilor e inceputa (pe Mac) |
| Test final pe test-dev | nefacut |

## Antrenarile complete (RTX 4070 SUPER)

| Model | Configurare | Date de antrenare | Epoci | Durata | Ponderi |
|---|---|---|---|---|---|
| baseline | `configs/train.yaml` | 6471 imagini intregi | 97 (oprit de `patience`), cea mai buna: 77 | 1,2 ore | `weights/yolo26n_baseline_640.pt` |
| pe felii | `configs/train_tiled.yaml` | 46 483 felii si imagini intregi | 100, cea mai buna: 82 | 7,7 ore | `weights/yolo26n_tiled.pt` |
| pe felii + CLAHE uniform | `configs/train_tiled.yaml` pe date cu CLAHE | idem, dupa CLAHE pe toate imaginile | 100 | ~7,7 ore | `weights/yolo26n_tiled_clahe_uniform.pt` |
| pe felii + CLAHE adaptiv | `configs/train_tiled.yaml` pe date cu CLAHE | idem, dupa CLAHE doar pe imaginile `dark` si `low_contrast` | 100, cea mai buna: 79 | ~8 ore (oprita o data si reluata) | `weights/yolo26n_tiled_clahe_adaptiv.pt` |

Seturile CLAHE ale acestor antrenari au fost facute cu o versiune anterioara a `pi-clahe`, cu pragurile de grupa calculate ca percentila 15 pe train (luminozitate 61,6, contrast 36,0), nu cu pragurile fixe 70 si 35 din codul actual. Pentru CLAHE uniform nu conteaza, pentru ca acolo toate imaginile primesc CLAHE. Pentru CLAHE adaptiv conteaza: setul de imagini imbunatatite difera putin de cel pe care l-ar alege codul actual.

### Etapa 1: cum impartim imaginea

| Varianta | Antrenare | `--mode` | mAP50 | mAP50-95 | ms/imagine* |
|---|---|---|---|---|---|
| Baseline | imagini intregi | `full` | 0,307 | 0,173 | 16 |
| Baseline + grila | imagini intregi | `slice` | 0,392 | 0,223 | 69 |
| Model pe felii, imagine intreaga | felii | `full` | 0,327 | 0,189 | 17 |
| **Model pe felii + grila** | felii | `slice` | **0,483** | **0,292** | 60 |
| Model pe felii + ferestre adaptive | felii | `adaptive` | 0,502 | 0,304 | 170 |

\* Masurat in timp ce placa video rula o alta antrenare, deci timpii absoluti sunt umflati; conteaza doar comparatia dintre randuri. Cu placa libera, `full` dadea ~8–10 ms, iar `slice` ~35–45 ms.

Ce arata tabelul:

- **Felierea la testare ajuta si fara reantrenare:** +0,085 mAP50 (randul 2 fata de randul 1).
- **Antrenarea pe felii adauga inca +0,091** peste asta (randul 4 fata de randul 2).
- **Antrenarea pe felii, singura, ajuta putin:** +0,020 (randul 3 fata de randul 1). Castigul vine din combinatia „antrenat pe felii, testat pe felii”.
- **Costul este timpul:** grila e de aproximativ 4 ori mai lenta decat imaginea intreaga.

Configuratia selectata pentru etapele urmatoare: **model pe felii + grila**.

### Pe clase (mAP50-95)

| Clasa | Baseline (`full`) | Model pe felii + grila | Diferenta |
|---|---|---|---|
| pedestrian | 0,154 | 0,289 | +0,135 |
| people | 0,092 | 0,182 | +0,090 |
| bicycle | 0,031 | 0,138 | +0,107 |
| car | 0,495 | 0,602 | +0,107 |
| van | 0,202 | 0,329 | +0,127 |
| truck | 0,175 | 0,298 | +0,123 |
| tricycle | 0,110 | 0,211 | +0,101 |
| awning-tricycle | 0,044 | 0,103 | +0,059 |
| bus | 0,274 | 0,477 | +0,203 |
| motor | 0,159 | 0,293 | +0,134 |

Toate clasele cresc. Bicycle creste de aproximativ 4,5 ori, iar pedestrian aproape se dubleaza. Cresterea la clasele cu obiecte mici este in acord cu ipoteza de la care a pornit felierea; nu am masurat insa separat pe marimi de obiecte, deci nu putem spune cat din castig vine de la ele.

### Pe grupe de iluminare (mAP50)

Grupele sunt calculate de `pi-stats` cu pragurile 70 si 35:

| Subset | dark | low_contrast | normal |
|---|---|---|---|
| val | 13 | 104 | 431 |
| test | 501 | 77 | 1032 |

| Varianta | dark (13) | low_contrast (104) | normal (431) |
|---|---|---|---|
| Baseline (`full`) | 0,401 | 0,307 | 0,303 |
| Baseline + grila | 0,448 | 0,370 | 0,391 |
| Model pe felii + grila | 0,573 | 0,503 | 0,479 |
| Model pe felii + adaptive | 0,565 | 0,524 | 0,498 |

**Val are doar 13 imagini intunecate.** Un mAP calculat pe 13 imagini se schimba mult de la o imagine la alta, deci coloana `dark` nu poate sustine nicio concluzie. Pentru etapa 2 (CLAHE), asta e o problema reala: efectul pe imaginile de noapte se va vedea bine abia pe test, care are 501.

Imaginile cu contrast slab nu ies mai prost decat cele normale. Asta nu inseamna ca iluminarea slaba ajuta: grupele difera si prin scena si prin numarul de obiecte.

### Etapa 2: CLAHE

Fiecare model CLAHE este evaluat pe val cu acelasi CLAHE aplicat (`VisDrone-clahe-uniform`, respectiv `VisDrone-clahe-adaptiv`), iar grupele sunt cele din `datasets/VisDrone/stats.csv`, calculate pe imaginile originale.

| Model (`slice`) | mAP50 | mAP50-95 | dark (13) | low_contrast (104) | normal (431) |
|---|---|---|---|---|---|
| pe felii, fara CLAHE | 0,483 | 0,292 | 0,573 | 0,503 | 0,479 |
| pe felii + CLAHE uniform | 0,482 | 0,289 | 0,571 | 0,494 | 0,480 |
| pe felii + CLAHE adaptiv | 0,484 | 0,292 | 0,562 | 0,468 | 0,483 |

Cu imaginea intreaga (`full`), cele trei modele dau mAP50 0,327, 0,325 si 0,326.

Ce arata tabelul:

- **Pe val, CLAHE nu schimba rezultatul.** Diferentele pe tot setul sunt de cel mult 0,002, adica in zgomotul dintre doua antrenari.
- **Pe imaginile slab contrastate, CLAHE adaptiv iese chiar mai prost** (0,468 fata de 0,503), desi acolo ar fi trebuit sa ajute. Pe 104 imagini, diferenta poate fi si intamplatoare.
- **Grupa `dark` are doar 13 imagini**, deci nu spune nimic. Efectul pe imaginile de noapte se poate masura doar pe test, care are 501.

Configuratia selectata ramane **model pe felii + grila, fara CLAHE**; CLAHE adaptiv va fi verificat o data pe test, la final, pentru grupa `dark`.

### Ferestre adaptive pe modelul pe felii

Cu parametrii impliciti din `configs/adaptive.yaml`: mAP50 0,502, fata de 0,483 la grila (+0,019), dar cu 10,2 ferestre pe imagine, care acopera impreuna de 4,5 ori suprafata imaginii, deci aproape de 3 ori mai lent. Aceeasi concluzie ca pe modelul de pe Mac de mai jos: metoda adaptiva detecteaza putin mai bine, dar nu economiseste timp fara reglarea parametrilor.

O varianta anterioara a ferestrelor adaptive, cu ferestre fixe de 640 px puse peste fiecare grup, a dat pe acelasi model mAP50 0,490 cu 5,5 ferestre pe imagine, iar varianta „adaptive + grila” 0,507. Ambele au fost scoase din cod la unirea cu `main`; valorile sunt pastrate aici pentru comparatie.

### Cum reproduci tabelele

```bash
G="--groups datasets/VisDrone/stats.csv"
uv run pi-stats
uv run pi-eval weights/yolo26n_baseline_640.pt --mode full     $G --out logs/eval/baseline_full.json
uv run pi-eval weights/yolo26n_baseline_640.pt --mode slice    $G --out logs/eval/baseline_slice.json
uv run pi-eval weights/yolo26n_tiled.pt        --mode full     $G --out logs/eval/tiled_full.json
uv run pi-eval weights/yolo26n_tiled.pt        --mode slice    $G --out logs/eval/tiled_slice.json
uv run pi-eval weights/yolo26n_tiled.pt        --mode adaptive $G --out logs/eval/tiled_adaptive.json
uv run pi-eval weights/yolo26n_tiled_clahe_uniform.pt --mode slice --data datasets/VisDrone-clahe-uniform/data.yaml $G
uv run pi-eval weights/yolo26n_tiled_clahe_adaptiv.pt --mode slice --data datasets/VisDrone-clahe-adaptiv/data.yaml $G
```

## Preliminar: modelul antrenat pe Mac (10 epoci)

Model: YOLO26n, 10 epoci pe `train`, imagini intregi de 640 px (`runs/yolo26n_mac`). Evaluare pe `val` (548 de imagini), pe imaginea intreaga, cu `pi-eval`, prag `conf 0.001`. Rezultatele complete sunt in `results/mac_*.json`.

Sunt rezultate preliminare: modelul e antrenat de 10 ori mai putin decat cel din metodologie, iar varianta „grila fixa” foloseste tot modelul pe imagini intregi, nu cel antrenat pe felii.

| Varianta | `--mode` | mAP50-95 | mAP50 | ms / imagine (Mac, MPS) | Pixeli procesati |
|---|---|---|---|---|---|
| Baseline | `full` | 0.113 | 0.202 | 10 | 1.00x |
| Grila fixa | `slice` | 0.174 | 0.305 | 60 | ≈3.4x |
| SAHI | `sahi` | 0.166 | 0.298 | 103 | ≈3.4x |
| Ferestre adaptive | `adaptive` | **0.185** | **0.327** | 154 | 5.66x |

mAP50 pe grupe:

| Varianta | dark (13) | low_contrast (104) | normal (431) |
|---|---|---|---|
| `full` | 0.300 | 0.217 | 0.196 |
| `slice` | 0.398 | 0.299 | 0.304 |
| `sahi` | 0.380 | 0.295 | 0.296 |
| `adaptive` | 0.408 | 0.316 | 0.326 |

### Ce arata

- Felierea creste mAP50 cu aproximativ 50% fata de baseline (0.202 → 0.305), confirmand ca obiectele mici sunt problema principala.
- Fuziunea proprie (`slice`) iese putin peste SAHI, cu aceleasi ferestre, si e mai rapida.
- Ferestrele adaptive au cel mai bun mAP, dar cu parametrii din `configs/adaptive.yaml` proceseaza **mai multi** pixeli decat grila (in medie 10.5 ferestre de cel putin 640 px, care se suprapun), deci nu economisesc timp. Castigul de precizie vine probabil din ferestrele centrate pe grupurile de obiecte. Pentru economie de timp trebuie reglate `dilate_kernel` si `min_size`, ca ferestrele sa fie mai putine.
- Grupa `dark` are doar 13 imagini pe val, deci valorile ei nu sunt concludente.

### Reglarea ferestrelor adaptive

Ferestrele adaptive pot fi reglate spre precizie sau spre cost. Toate randurile sunt pe tot `val`, cu acelasi model:

| Setare | mAP50 | mAP50-95 | ms / imagine | Ferestre / imagine | Pixeli procesati |
|---|---|---|---|---|---|
| grila fixa (`slice`), referinta | 0.305 | 0.174 | 60 | 6 (la 1360×765) | ≈3.4x |
| implicit (`dilate_kernel=15`, `min_size=640`) | **0.327** | **0.185** | 154 | 10.5 | 5.66x |
| `dilate_kernel=31` | 0.305 | 0.171 | 84 | 5.5 | 3.45x |
| `dilate_kernel=31 min_size=480` | 0.301 | 0.168 | 74 | 5.5 | **2.56x** |

- Cu setarile implicite, metoda adaptiva bate grila cu 0.02 mAP50, dar proceseaza mai multi pixeli.
- Cu ferestre mai putine si mai mici, ajunge la aceeasi precizie ca grila cu cu 25% mai putini pixeli. Timpul ramane totusi mai mare, pentru ca se adauga detectia preliminara, iar ferestrele de marimi diferite se proceseaza mai putin eficient decat cele egale ale grilei.
- Pe primele 100 de imagini din `val`, `min_size=480` parea sa dea mAP50 0.350. Pe tot setul a scazut la 0.301. Primele 100 de imagini vin din cateva secvente video, deci nu sunt reprezentative, iar reglajele trebuie verificate pe tot `val`.

### Figuri

Pasii metodei adaptive pe o imagine de noapte din `val` (`0000117_02708_d_0000090.jpg`), cu setarile implicite: 62 de casete preliminare, 15 ferestre, 150 de detectii finale.

![Pasii ferestrelor adaptive](../results/figuri/ferestre_adaptive_pasi.png)

![Fara ferestre comparat cu ferestrele adaptive](../results/figuri/ferestre_adaptive_comparatie.png)

Pixelii procesati pentru `slice` si `sahi` sunt calculati pentru imaginile de 1360×765 (6 ferestre de 640 px plus imaginea intreaga); evaluatorul ii masoara direct doar la `adaptive`.
