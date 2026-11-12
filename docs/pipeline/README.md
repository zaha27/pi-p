# Pipeline-ul proiectului, pas cu pas

Acest folder explică fiecare etapă a proiectului: ce problemă rezolvă, cum funcționează, unde e codul și cu ce comandă se rulează. Fiecare etapă are fișierul ei. Citite în ordine, ele urmează drumul unei imagini de la descărcare până la tabelul de rezultate.

## Imaginea de ansamblu

Proiectul are două părți care se întâlnesc la model:

- **pregătirea și antrenarea** se fac o singură dată și produc un fișier de greutăți (`best.pt`);
- **detecția și evaluarea** folosesc acel fișier pe imagini noi.

```mermaid
flowchart TB
    subgraph PREG["Pregătire și antrenare (o singură dată)"]
        direction TB
        P1["`**1. Pregătirea datelor**
pi-prepare, pi-check, pi-eda`"]
        P2["`**2. Analiza statistică**
pi-stats`"]
        P3["`**3. Îmbunătățirea contrastului**
pi-clahe (opțional)`"]
        P4["`**4. Felierea pe grilă**
pi-tile (opțional)`"]
        P5["`**5. Antrenarea**
pi-train`"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    M[("best.pt
modelul antrenat")]

    subgraph DET["Detecție și evaluare (pe fiecare imagine)"]
        direction TB
        D6["`**6. Detecția pe ferestre**
pe grilă sau pe ferestre adaptive (9)`"]
        D7["`**7. Fuziunea detecțiilor**
elimină bucățile și dublurile`"]
        D8["`**8. Evaluarea**
pi-eval: mAP și timp pe imagine`"]
        D6 --> D7 --> D8
    end

    P5 --> M --> D6
    P2 -. "grupele de iluminare" .-> D8
```

## Etapele

| Nr. | Etapa | Ce face, pe scurt | Cod | Comandă |
|---|---|---|---|---|
| 1 | [Pregătirea datelor](01-pregatirea-datelor.md) | descarcă VisDrone, îl convertește în format YOLO, îl verifică | `prepare.py`, `check.py`, `eda.py` | `pi-prepare`, `pi-check`, `pi-eda` |
| 2 | [Analiza statistică](02-analiza-statistica.md) | măsoară luminozitatea și contrastul, împarte imaginile pe grupe | `stats.py` | `pi-stats` |
| 3 | [Îmbunătățirea contrastului](03-imbunatatirea-contrastului.md) | aplică CLAHE, pe toate imaginile sau doar pe cele care îl cer | `enhance.py` | `pi-clahe` |
| 4 | [Felierea pe grilă](04-felierea-pe-grila.md) | taie imaginea în ferestre suprapuse de 640 px | `slicing.py`, `tile.py` | `pi-tile` |
| 5 | [Antrenarea](05-antrenarea.md) | fine-tuning YOLO26n pe imagini întregi sau pe felii | `train.py`, `config.py` | `pi-train` |
| 6 | [Detecția pe ferestre](06-detectia-pe-ferestre.md) | rulează modelul pe fiecare fereastră și mută chenarele în imaginea întreagă | `predict.py` | `pi-predict` |
| 7 | [Fuziunea detecțiilor](07-fuziunea-detectiilor.md) | elimină obiectele tăiate și dublurile dintre ferestre | `predict.py` | (parte din 6) |
| 8 | [Evaluarea](08-evaluarea.md) | compară detecțiile cu etichetele și calculează mAP | `evaluate.py`, `metrics.py` | `pi-eval` |
| 9 | [Ferestrele adaptive](09-ferestre-adaptive.md) | pune ferestre doar unde sunt obiecte | `adaptive.py` | `pi-eval --mode adaptive` |

Rezultatele măsurate până acum sunt în [../rezultate.md](../rezultate.md), iar planul experimentelor în [../benchmarkuri.md](../benchmarkuri.md). Un rezumat pe o singură pagină, cu legătura dintre comenzi, este în [../pipeline.md](../pipeline.md), iar codul explicat funcție cu funcție în [../cod.md](../cod.md).

## Cum depind modulele unele de altele

Săgeata înseamnă „folosește”. Modulele de jos nu știu nimic despre cele de sus, deci pot fi citite și testate separat.

```mermaid
flowchart BT
    data["data.py
formatul YOLO pe disc"]
    slicing["slicing.py
grila de ferestre"]
    stats["stats.py
indicatori și grupe"]
    enhance["enhance.py
CLAHE"]
    predict["predict.py
drop_cut, to_image, merge, sliced_predict"]
    adaptive["adaptive.py
ferestre adaptive"]
    metrics["metrics.py
potrivire și mAP"]
    prepare["prepare.py"]
    tile["tile.py"]
    evaluate["evaluate.py"]
    train["train.py"]
    config["config.py"]

    prepare --> data
    stats --> data
    enhance --> stats
    enhance --> data
    tile --> slicing
    tile --> data
    predict --> slicing
    adaptive --> predict
    adaptive --> config
    evaluate --> predict
    evaluate --> adaptive
    evaluate --> metrics
    evaluate --> stats
    train --> config
```

Două module din `src/pi_p/` nu apar în schemă pentru că țin de aplicația web, nu de pipeline: `analytics.py` (statistici de trafic) și `scoremap.py` (harta de scoruri a detectorului). Serverul și dashboard-ul sunt în `infrastructure/` și sunt descrise în [../uml.md](../uml.md).

## Toate comenzile, în ordine

```bash
uv sync                                   # instalează dependențele

# 1. date
uv run pi-prepare                         # descarcă și convertește VisDrone
uv run pi-check                           # verifică setul de date
uv run pi-eda                             # grafice despre obiecte

# 2. grupe de iluminare
uv run pi-stats

# etapa 1 din plan: imagini întregi vs. felii
uv run pi-train configs/train.yaml        # baseline
uv run pi-tile                            # felii pentru antrenare
uv run pi-train configs/train_tiled.yaml  # model pe felii

# evaluare, mereu pe imaginile întregi din val
G="--groups datasets/VisDrone/stats.csv"
uv run pi-eval weights/yolo26n_baseline_640.pt --mode full  $G
uv run pi-eval weights/yolo26n_baseline_640.pt --mode slice $G
uv run pi-eval weights/yolo26n_baseline_640.pt --mode sahi  $G
uv run pi-eval weights/yolo26n_tiled.pt        --mode slice $G

# etapa 2 din plan: CLAHE (cere reantrenare)
uv run pi-clahe --mode uniform
uv run pi-tile --src datasets/VisDrone-clahe --dst datasets/VisDrone-clahe-tiled
uv run pi-train configs/train_tiled.yaml data=datasets/VisDrone-clahe-tiled/data.yaml name=yolo26n_tiled_clahe_uniform
uv run pi-eval runs/yolo26n_tiled_clahe_uniform/weights/best.pt --mode slice --data datasets/VisDrone-clahe/data.yaml $G

# etapa 3 din plan: ferestre adaptive (fără reantrenare)
uv run pi-eval weights/yolo26n_tiled.pt --mode adaptive $G
```

## Testele

Fiecare etapă are teste în `tests/`, care rulează fără date reale și fără placă video:

```bash
uv run pytest
```

| Fișier de test | Ce verifică |
|---|---|
| `test_prepare.py` | conversia adnotărilor și acoperirea regiunilor ignorate |
| `test_check_tile.py` | verificarea setului de date și felierea pentru antrenare |
| `test_enhance.py` | indicatorii, grupele și CLAHE |
| `test_slicing.py` | grila de ferestre, `drop_cut` și `merge` |
| `test_adaptive.py` | gruparea obiectelor și ferestrele adaptive |
| `test_metrics.py` | potrivirea detecțiilor și calculul mAP |
| `test_evaluate.py` | evaluatorul, pe toate modurile și pe grupe |
| `test_adaptive_figure.py` | figurile ferestrelor adaptive |
