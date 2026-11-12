# Pipeline-ul proiectului

Documentul arata cum se leaga comenzile `pi-*` intre ele: ce primeste fiecare, ce produce si ce varianta din experimente acopera. Motivatia metodelor e in [Arhitectura_si_metodologie.md](Arhitectura_si_metodologie.md), iar ordinea experimentelor in [benchmarkuri.md](benchmarkuri.md).

## 1. De la date la rezultate

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'15px','lineColor':'#4B5563'}}}%%
flowchart TB
    RAW["`**VisDrone2019-DET**
adnotari in formatul original`"]
    PREP["`**pi-prepare**
format YOLO, zonele ignorate acoperite cu gri`"]
    CHECK["`**pi-check**
imagini corupte, etichete invalide`"]
    EDA["`**pi-eda**
clase, marimi, densitate`"]
    STATS["`**pi-stats**
luminozitate si contrast → stats.csv`"]
    CLAHE["`**pi-clahe**
uniform sau adaptiv`"]
    TILE["`**pi-tile**
felii de 640 px, suprapunere 20%`"]
    TRAIN["`**pi-train**
fine-tuning YOLO26n → best.pt`"]
    EVAL["`**pi-eval**
mAP total, pe clase si pe grupe`"]

    RAW --> PREP --> CHECK
    CHECK --> EDA
    CHECK --> STATS
    STATS -- "grupele" --> CLAHE
    CHECK --> TRAIN
    CHECK --> TILE --> TRAIN
    CLAHE -. "etapa 2" .-> TILE
    CLAHE -. "etapa 2" .-> TRAIN
    TRAIN --> EVAL
    STATS -- "--groups" --> EVAL

    classDef dat fill:#F3F4F6,stroke:#374151,stroke-width:2px,color:#111827
    classDef cmd fill:#DBEAFE,stroke:#1D4ED8,stroke-width:2px,color:#111827
    classDef trn fill:#DCFCE7,stroke:#15803D,stroke-width:2px,color:#111827
    classDef eva fill:#FEF3C7,stroke:#B45309,stroke-width:2px,color:#111827
    class RAW dat
    class PREP,CHECK,EDA,STATS,CLAHE,TILE cmd
    class TRAIN trn
    class EVAL eva
```

Fiecare pas scrie un set de date nou in `datasets/` si nu il modifica pe cel de intrare, cu exceptia lui `pi-prepare`, care converteste datele pe loc:

| Set de date | Produs de | Folosit pentru |
|---|---|---|
| `datasets/VisDrone` | `pi-prepare` | antrenarea pe imagini intregi si toate evaluarile |
| `datasets/VisDrone-tiled` | `pi-tile` | antrenarea pe felii |
| `datasets/VisDrone-clahe` | `pi-clahe --mode uniform` | etapa 2, CLAHE pe toate imaginile |
| `datasets/VisDrone-clahe-adaptive` | `pi-clahe --mode adaptive` | etapa 2, CLAHE doar pe grupele `dark` si `low_contrast` |

## 2. Ce face `pi-eval` cu o imagine

Toate modurile primesc aceeasi imagine intreaga si aceleasi etichete, deci difera doar felul in care se obtin detectiile. Calculul mAP-ului e comun ([metrics.py](../src/pi_p/metrics.py)).

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'15px','lineColor':'#4B5563'}}}%%
flowchart LR
    IMG["imaginea intreaga"]
    FULL["`**full**
o trecere a modelului`"]
    SLICE["`**slice**
imaginea + ferestrele grilei`"]
    SAHI["`**sahi**
la fel, prin biblioteca SAHI`"]
    ADA["`**adaptive**
detectie preliminara → ferestre doar unde sunt obiecte`"]
    SHIFT["`**to_image**
casetele taiate de marginea ferestrei se arunca, restul se muta in coordonatele imaginii`"]
    MERGE["`**merge**
dublurile de aceeasi clasa se elimina dupa IOS`"]
    MAP["`**mean_ap**
acelasi plafon de detectii pentru toate modurile`"]

    IMG --> FULL --> MAP
    IMG --> SLICE --> SHIFT
    IMG --> ADA --> SHIFT
    SHIFT --> MERGE --> MAP
    IMG --> SAHI --> MAP

    classDef dat fill:#F3F4F6,stroke:#374151,stroke-width:2px,color:#111827
    classDef var fill:#DBEAFE,stroke:#1D4ED8,stroke-width:2px,color:#111827
    classDef fus fill:#DCFCE7,stroke:#15803D,stroke-width:2px,color:#111827
    classDef eva fill:#FEF3C7,stroke:#B45309,stroke-width:2px,color:#111827
    class IMG dat
    class FULL,SLICE,SAHI,ADA var
    class SHIFT,MERGE fus
    class MAP eva
```

SAHI face aceiasi pasi in interiorul bibliotecii, cu propria fuziune (`GREEDYNMM` dupa IOS).

Ferestrele adaptive ([adaptive.py](../src/pi_p/adaptive.py)) se construiesc din detectiile preliminare astfel:

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'15px','lineColor':'#4B5563'}}}%%
flowchart LR
    P["`**detectie preliminara**
imaginea micsorata, prag prelim_conf`"]
    M["`**masca**
casetele desenate pe o masca de mask_size`"]
    D["`**dilatare**
obiectele apropiate se unesc`"]
    R["`**dreptunghiuri**
componente conexe`"]
    W["`**ferestre**
cu margine, minim min_size, in interiorul imaginii`"]
    P --> M --> D --> R --> W
    classDef var fill:#DBEAFE,stroke:#1D4ED8,stroke-width:2px,color:#111827
    class P,M,D,R,W var
```

Parametrii sunt in [configs/adaptive.yaml](../configs/adaptive.yaml) si se pot schimba din linia de comanda cu `key=value`.

## 3. Comenzile

| Comanda | Modul | Intrare | Iesire | Optiuni importante |
|---|---|---|---|---|
| `pi-prepare` | [prepare.py](../src/pi_p/prepare.py) | arhivele VisDrone (descarcate automat) | `datasets/VisDrone`, `data.yaml` | `--splits` |
| `pi-check` | [check.py](../src/pi_p/check.py) | setul de date | raport in consola, cod 1 la erori | `--splits` |
| `pi-eda` | [eda.py](../src/pi_p/eda.py) | un split | `eda_out/summary.json` si grafice | `--imgsz 640 1024` |
| `pi-stats` | [stats.py](../src/pi_p/stats.py) | toate split-urile | `stats.csv`, `stats.png` | `--dark 70`, `--low-contrast 35` |
| `pi-clahe` | [enhance.py](../src/pi_p/enhance.py) | setul de date si `stats.csv` | set nou, cu imagini imbunatatite | `--mode uniform\|adaptive`, `--clip`, `--grid` |
| `pi-tile` | [tile.py](../src/pi_p/tile.py) | setul de date | set nou, cu felii | `--size`, `--overlap`, `--min-visibility`, `--empty` |
| `pi-train` | [train.py](../src/pi_p/train.py) | un fisier din `configs/` | `runs/<name>/weights/best.pt` | `key=value` peste config |
| `pi-eval` | [evaluate.py](../src/pi_p/evaluate.py) | `best.pt` si setul de date | mAP in consola, JSON cu `--out` | `--mode`, `--groups`, `--limit` |
| `pi-predict` | [predict.py](../src/pi_p/predict.py) | imagine, folder sau video | imagini cu detectiile desenate | `--slice` |
| `python -m pi_p.adaptive_figure` | [adaptive_figure.py](../src/pi_p/adaptive_figure.py) | o imagine | figura cu pasii ferestrelor adaptive | `--comparison-out` |

Modurile din `pi-eval`:

| `--mode` | Ce ruleaza | Varianta din experimente |
|---|---|---|
| `val` | `model.val()` din Ultralytics, cu `--coco` pentru modelul preantrenat | verificare rapida |
| `full` | o trecere pe imaginea intreaga | baseline |
| `slice` | imaginea intreaga si grila de ferestre, fuziune proprie | feliere pe grila fixa |
| `sahi` | acelasi lucru, prin biblioteca SAHI | SAHI |
| `adaptive` | detectie preliminara si ferestre adaptive | ferestre adaptive |

## 4. Reteta experimentelor

Toate rularile de comparatie se fac pe `val`, cu pragul implicit `--conf 0.001`, si salveaza rezultatul in JSON:

```bash
uv run pi-stats
uv run pi-train configs/train.yaml
uv run pi-tile
uv run pi-train configs/train_tiled.yaml

G=datasets/VisDrone/stats.csv
uv run pi-eval runs/yolo26n_640/weights/best.pt   --mode full  --groups $G --out results/full.json
uv run pi-eval runs/yolo26n_640/weights/best.pt   --mode sahi  --groups $G --out results/sahi.json
uv run pi-eval runs/yolo26n_tiled/weights/best.pt --mode slice --groups $G --out results/slice.json
uv run pi-eval runs/yolo26n_640/weights/best.pt   --mode adaptive --groups $G --out results/adaptive.json
```

Pentru etapa 2 se ruleaza `pi-clahe --mode uniform` si `pi-clahe --mode adaptive`, apoi configuratia castigatoare din etapa 1 se reantreneaza pe fiecare set nou (`pi-train <config> data=datasets/VisDrone-clahe/data.yaml name=...`) si se evalueaza la fel, cu `--data` spre acelasi set imbunatatit, ca modelul sa vada la test aceleasi imagini ca la antrenare.

`--limit 20` ruleaza doar primele 20 de imagini, util ca verificare inainte de rularea completa.
