# 1. Pregătirea datelor

**Cod:** `src/pi_p/prepare.py`, `check.py`, `eda.py`, `data.py` · **Comenzi:** `pi-prepare`, `pi-check`, `pi-eda`

## De ce e nevoie de etapa asta

VisDrone vine în formatul lui propriu, iar Ultralytics YOLO citește alt format. În plus, VisDrone are regiuni marcate ca „ignorate”: zone cu obiecte prea mici sau prea dese, pe care adnotatorii nu le-au etichetat unul câte unul. Dacă le lăsăm în imagine, modelul vede acolo mașini fără etichetă și învață că „nu sunt obiecte”, iar la evaluare e penalizat dacă le detectează.

## Schema

```mermaid
flowchart LR
    A["Arhivele VisDrone
train, val, test-dev"] --> B["parse
text → tabel cu 8 coloane"]
    B --> C["convert
separă obiectele de regiunile ignorate"]
    C --> D["mask_ignored
acoperă cu gri regiunile ignorate"]
    C --> E["write_labels
etichete în format YOLO"]
    D --> F[("datasets/VisDrone
images/, labels/, data.yaml")]
    E --> F
    F --> G["pi-check
caută greșeli"]
    F --> H["pi-eda
grafice despre obiecte"]
```

## Cele două formate

O linie dintr-un fișier de adnotări VisDrone descrie un obiect:

```
x, y, lățime, înălțime, scor, categorie, trunchiere, ocluzie
684, 8, 273, 116, 0, 0, 0, 0
406, 119, 265, 70, 1, 4, 0, 0
```

- `(x, y)` este colțul stânga-sus al chenarului, în pixeli;
- `scor` 0 înseamnă „ignoră acest chenar”;
- `categorie` 1–10 sunt clasele reale; 0 este „regiune ignorată”, 11 este „altele”.

Aceeași mașină (a doua linie), în format YOLO, într-o imagine de 1360 × 765:

```
3 0.395956 0.201307 0.194853 0.091503
```

- prima valoare este clasa, numărată de la 0 (4 − 1 = 3, adică `car`);
- urmează centrul chenarului și dimensiunile lui, împărțite la lățimea și înălțimea imaginii.

Calculul pentru centrul pe orizontală: (406 + 265 / 2) / 1360 = 0.3960.

## Ce se întâmplă cu fiecare adnotare

Funcția `convert` împarte liniile în trei categorii:

| Condiție | Ce devine |
|---|---|
| scor ≠ 0 și categorie între 1 și 10 | obiect păstrat, scris în etichete |
| scor = 0, categorie 0 sau categorie 11 | regiune ignorată, acoperită cu gri |
| chenar cu arie zero după tăierea la marginile imaginii | aruncat |

Funcția `mask_ignored` vopsește regiunile ignorate cu gri (valoarea 114, aceeași cu care YOLO completează marginile imaginii). Dacă un obiect păstrat se află într-o regiune ignorată, pixelii lui rămân vizibili.

## Verificarea și analiza

`pi-check` trece prin tot setul și raportează imagini corupte, imagini fără etichetă, clase inexistente, chenare cu arie zero sau în afara imaginii și linii duplicate. Se rulează și după feliere, pentru că acolo se scriu etichete noi.

`pi-eda` face grafice despre date: câte obiecte are fiecare clasă, câte obiecte sunt pe imagine și, cel mai important, cât de mari sunt obiectele așa cum le vede modelul. O imagine de 1920 px micșorată la 640 px își micșorează și obiectele de 3 ori, iar multe ajung sub 16 px. Acesta e motivul pentru felierea din etapa 4.

## Rezultatul

```
datasets/VisDrone/
    images/train/   6471 imagini
    images/val/      548 imagini
    images/test/    1610 imagini
    labels/...      câte un .txt pentru fiecare imagine
    annotations/... adnotările originale, păstrate
    data.yaml       ce citește Ultralytics
```

## Comenzi

```bash
uv run pi-prepare          # descarcă (~2 GB) și convertește
uv run pi-check            # trebuie să afișeze OK pentru train și val
uv run pi-eda              # scrie graficele în eda_out/
```
