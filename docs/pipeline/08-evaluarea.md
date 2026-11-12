# 8. Evaluarea

**Cod:** `src/pi_p/evaluate.py`, `src/pi_p/metrics.py` · **Comanda:** `pi-eval`

## Ce masuram

Vrem un singur numar care spune cat de bine detecteaza o varianta, ca sa putem compara variantele intre ele. Numarul este **mAP** (mean Average Precision). Pe langa el masuram **timpul pe imagine**, pentru ca felierea face detectia mai buna, dar si mai lenta.

## Regula de baza: toate variantele sunt evaluate la fel

- pe **imaginile intregi** din val, dupa fuziunea detectiilor;
- cu **etichetele imaginii intregi** ca referinta;
- cu **acelasi prag de incredere**, mic (0,001).

Daca am evalua modelul pe felii pe feliile lui, obiectele ar aparea mari si scorul ar iesi mai mare fara ca modelul sa fie mai bun. Pragul trebuie sa fie mic si acelasi pentru ca mAP se calculeaza coborand treptat pragul; o varianta evaluata cu prag mai mare ar pierde partea de jos a curbei.

## Schema

```mermaid
flowchart TB
    I["Imagine intreaga din val"] --> D["detect
in modul cerut: full, slice, sahi, adaptive"]
    D --> T["cronometru
ms pe imagine"]
    D --> M["match
care detectii sunt corecte?"]
    L["Etichetele imaginii"] --> M
    M --> A["pentru fiecare detectie:
corecta?, scor, clasa"]
    A --> S["mean_ap
AP pe clasa → mAP"]
    G["Grupele de la pi-stats"] --> S
    S --> R1["mAP pe tot setul"]
    S --> R2["mAP pe fiecare grupa"]
    S --> R3["mAP pe fiecare clasa"]
```

## De la detectii la mAP, in patru pasi

### 1. Cand e corecta o detectie (`metrics.match`)

O detectie este corecta daca are **aceeasi clasa** cu un obiect adnotat si chenarele lor se suprapun suficient:

```
IoU = aria intersectiei / aria reuniunii   ≥ prag
```

Fiecare obiect adnotat poate fi „folosit” o singura data. Daca doua detectii nimeresc aceeasi masina, doar cea cu IoU mai mare e corecta; cealalta e o alarma falsa.

### 2. Precizie si recall

```
precizie = detectii corecte / toate detectiile        cat din ce am zis e adevarat
recall   = detectii corecte / toate obiectele reale   cat din ce exista am gasit
```

Cele doua trag in directii opuse. Cu un prag de incredere mare, modelul spune putine lucruri, dar sigure (precizie mare, recall mic). Cu un prag mic, gaseste mai mult, dar si greseste mai mult.

### 3. AP: aria de sub curba precizie–recall

Ordonam toate detectiile unei clase dupa scor si coboram pragul treptat. La fiecare pas notam precizia si recall-ul, ceea ce da o curba. **AP** (Average Precision) este aria de sub ea: 1 pentru un detector perfect, 0 pentru unul care nu nimereste nimic. AP nu depinde de un prag ales de noi, pentru ca le acopera pe toate.

### 4. mAP: media pe clase

**mAP** este media AP-urilor celor 10 clase. Fiecare clasa cantareste la fel, deci o clasa rara si grea (awning-tricycle) trage media in jos la fel de mult cat o ridica una frecventa si usoara (car).

Raportam doua variante:

| Metrica | Prag IoU | Ce recompenseaza |
|---|---|---|
| mAP50 | 0,5 | gasirea obiectului, chiar cu un chenar aproximativ |
| mAP50-95 | media pe 0,50, 0,55, …, 0,95 | si potrivirea exacta a chenarului |

## Evaluarea pe grupe

Cu `--groups datasets/VisDrone/stats.csv` (tabelul scris de `pi-stats`), aceleasi calcule se repeta separat pentru imaginile `dark`, `low_contrast` si `normal`. Asa vedem daca o metoda, de exemplu CLAHE, ajuta imaginile grele chiar cand media pe tot setul abia se schimba.

Grupele trebuie sa vina mereu din imaginile **originale** (`datasets/VisDrone/stats.csv`), si atunci cand evaluam un set cu CLAHE. Altfel, dupa CLAHE, imaginile intunecate n-ar mai parea intunecate si grupele n-ar mai fi aceleasi de la o varianta la alta.

## Timpul pe imagine

Timpul se masoara in jurul apelului `detect`, deci include toate rularile modelului si fuziunea, dar nu citirea imaginii de pe disc. Inainte de masurare, primele 3 imagini sunt rulate o data in plus, necronometrat, pentru ca acolo se incarca modelul pe placa video.

## De ce valorile difera de cele din antrenare

`pi-eval --mode full` nu da exact acelasi mAP ca `pi-eval --mode val`, care ruleaza validarea Ultralytics (`model.val`). Diferenta vine din felul in care imaginea e redimensionata si completata cu margini inainte de a intra in model, care nu e identic in cele doua cai. Potrivirea detectiilor cu etichetele este testata in `tests/test_metrics.py`.

Pentru comparatii conteaza ca **toate variantele trec prin aceeasi cale**, cea din `pi-eval --mode full/slice/sahi/adaptive`. Nu se compara un numar din aceasta cale cu unul din jurnalul de antrenare sau din `--mode val`.

## Rolul subseturilor

| Subset | La ce e folosit |
|---|---|
| train | doar antrenare |
| val | toate comparatiile si toate deciziile |
| test | o singura rulare, la final, dupa ce deciziile sunt inghetate |

Daca am alege varianta cea mai buna uitandu-ne la test, rezultatul raportat pe test ar fi prea optimist.

## Comenzi

```bash
uv run pi-eval weights/yolo26n_baseline_640.pt --mode full                        # baseline
uv run pi-eval weights/yolo26n_tiled.pt --mode slice                               # model pe felii, pe grila
uv run pi-eval weights/yolo26n_tiled.pt --mode slice --groups datasets/VisDrone/stats.csv --out logs/eval/nume.json
uv run pi-eval weights/yolo26n_tiled.pt --mode slice --split test                  # doar la final
uv run pi-eval yolo26n.pt --mode val --coco                                        # model COCO, fara fine-tuning
```
