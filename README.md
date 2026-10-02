# pi-p

Detecție de obiecte în imagini filmate din dronă ([VisDrone2019-DET](https://docs.ultralytics.com/datasets/detect/visdrone/), 10 clase) cu YOLO26, optimizat pentru obiecte mici.

## Structură

```
configs/      train.yaml (640), train_tiled.yaml (pe tile-uri), smoke.yaml (test rapid)
src/pi_p/     prepare, check, eda, tile, train, evaluate, predict, scoremap, analytics
src/pi_p/app/ server FastAPI: upload, streaming WebSocket, sesiune video
frontend/     dashboard Vite + TypeScript
tests/        teste pe un mini-dataset sintetic în formatul VisDrone
datasets/     generat de pi-prepare / pi-tile (ignorat în git)
runs/         rezultate (ignorat în git)
```

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Pe Windows/Linux se instalează `torch` cu CUDA 12.8, pe Mac varianta standard (MPS).

## Pipeline

```bash
uv run pi-prepare                 # descarcă (~2.3 GB) și convertește în format YOLO
uv run pi-check                   # verifică integritatea imaginilor și etichetelor
uv run pi-eda                     # statistici și grafice în eda_out/
uv run pi-tile                    # dataset pe tile-uri 640px în datasets/VisDrone-tiled
uv run pi-eval yolo26n.pt         # baseline: model COCO, neantrenat pe VisDrone
uv run pi-train configs/smoke.yaml
uv run pi-train                   # sau: pi-train configs/train_tiled.yaml
uv run pi-train configs/train.yaml imgsz=1024 batch=8 name=yolo26n_1024
uv run pi-eval runs/yolo26n_640/weights/best.pt
uv run pi-predict runs/yolo26n_tiled/weights/best.pt imagine.jpg --slice 640
```

`pi-prepare` face conversia proprie, nu pe cea din Ultralytics: zonele marcate „ignored region” și obiectele „others” sunt acoperite cu gri (114) în imagine, ca modelul să nu fie penalizat pentru obiecte reale neetichetate. Adnotările originale rămân în `datasets/VisDrone/annotations/`.

`pi-predict --slice N` rulează modelul pe ferestre N×N suprapuse plus pe imaginea întreagă. Elimină detecțiile tăiate de marginea ferestrei, apoi combină duplicatele după IOS.

## Dashboard

Dashboard-ul procesează un video din dronă și afișează live:
- **detecția**, cu ID și traiectorie pentru fiecare obiect; poți comuta pe harta de scoruri a detectorului;
- **sumarul**: obiecte în cadru, obiecte unice, flux mediu și timp de inferență, plus ponderea și evoluția fiecărei clase;
- **fluxul în timp**, stivuit pe clase;
- **câmpul de mișcare**, cu vectorii de viteză;
- **harta de congestie**, ponderată cu timpul de staționare, cu procentul de saturare.

Pragul de încredere și vizualizarea se pot schimba în timpul rulării. La final descarci video-ul adnotat și CSV-ul cu numărătoarea pe cadre.

```bash
cd frontend && npm install && npm run build && cd ..
uv run pi-app                     # http://127.0.0.1:8000
```

În dezvoltare rulezi `uv run pi-app` și, separat, `npm run dev --prefix frontend` (http://localhost:5173, cu proxy spre API).

Harta de congestie și vectorii de flux presupun o dronă care stă pe loc. Dacă drona se mișcă, mișcarea camerei se adună peste mișcarea obiectelor.

## Docker și Cloudflare

Pe PC-ul cu GPU NVIDIA, cu NVIDIA Container Toolkit instalat:

```bash
docker compose up -d --build                         # http://localhost:8000
```

Modelele antrenate din `runs/` sunt montate în container. Ca aplicația să fie accesibilă public prin Cloudflare Tunnel:

1. În Cloudflare Zero Trust creezi un tunel care trimite traficul domeniului spre `http://app:8000`.
2. Pui tokenul tunelului în `.env` (vezi `.env.example`).
3. Pornești aplicația împreună cu tunelul:

```bash
docker compose --profile tunnel up -d
```

Recomandare: pune aplicația în spatele Cloudflare Access, altfel oricine are link-ul poate încărca video-uri și folosi GPU-ul.

## Teste

```bash
uv run pytest -q
```
