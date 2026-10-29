# Arhitectura proiectului (UML)

Diagramele sunt scrise în Mermaid și se afișează direct pe GitHub.

1. [Deployment](#1-diagrama-de-deployment): unde rulează fiecare parte
2. [Componente](#2-diagrama-de-componente): modulele și dependențele dintre ele
3. [Clase](#3-diagrama-de-clase): clasele și funcțiile principale
4. [Secvență: sesiune de analiză](#4-diagrama-de-secvență-sesiune-de-analiză)
5. [Secvență: procesarea unui cadru](#5-diagrama-de-secvență-procesarea-unui-cadru)
6. [Activitate: pipeline-ul de date și antrenare](#6-diagrama-de-activitate-pipeline-ul-de-date-și-antrenare)
7. [Stări: interfața](#7-diagrama-de-stări-interfața)

## 1. Diagrama de deployment

```mermaid
flowchart LR
    subgraph Client["Dispozitivul utilizatorului"]
        B["Browser<br/>dashboard Vite + TypeScript"]
        GA["Google Authenticator<br/>generează coduri TOTP"]
    end

    subgraph CF["Cloudflare"]
        DNS["DNS<br/>pi.zaha.cc → tunel"]
        EDGE["Edge<br/>HTTPS, limită 100 MB/cerere"]
    end

    subgraph Host["Server: PC cu RTX 4070 Super sau Mac"]
        subgraph Docker["docker compose"]
            T["container tunnel<br/>cloudflared"]
            A["container app<br/>FastAPI + Uvicorn :8000"]
        end
        V[("videos/<br/>biblioteca")]
        R[("runs/<br/>modele antrenate")]
        W[("/tmp/pi-p<br/>upload-uri și rezultate")]
        GPU["GPU<br/>CUDA / MPS / CPU"]
    end

    B -- "HTTPS + WebSocket" --> EDGE
    DNS -.-> EDGE
    EDGE -- "tunel criptat (QUIC)" --> T
    T -- "http://app:8000" --> A
    A --- V
    A --- R
    A --- W
    A -- "PyTorch / Ultralytics" --> GPU
    GA -. "codul de 6 cifre, introdus manual" .-> B
```

Containerul `app` nu e expus direct în rețea: portul 8000 e legat doar de `127.0.0.1`. Din internet se ajunge numai prin tunel, iar orice cerere `/api/*` cere autentificare.

## 2. Diagrama de componente

```mermaid
flowchart TB
    subgraph FE["infrastructure/frontend/src (browser)"]
        main["main.ts<br/>controale, WebSocket, pauză"]
        dashboard["dashboard.ts<br/>desenează panourile"]
        charts["charts.ts<br/>sparkline, flux, câmp de mișcare"]
        library_ts["library.ts<br/>tabul Bibliotecă"]
        auth_ts["auth.ts<br/>ecranul de login"]
        api_ts["api.ts<br/>upload, modele"]
        dom["dom.ts"]
        types["types.ts<br/>mesajele serverului"]
    end

    subgraph APP["infrastructure/pi_p_server (server web)"]
        server["server.py<br/>rute HTTP, WebSocket, login"]
        session["session.py<br/>Session: un video procesat"]
        render["render.py<br/>desen cutii, hărți"]
        library_py["library.py<br/>listare, thumbnail"]
        auth_py["auth.py<br/>TOTP, sesiuni semnate"]
    end

    subgraph CORE["src/pi_p (logica de bază)"]
        analytics["analytics.py<br/>flux, congestie, statistici"]
        scoremap["scoremap.py<br/>harta de scoruri a detectorului"]
        predict["predict.py<br/>inferență pe bucăți"]
        slicing["slicing.py<br/>ferestre suprapuse"]
        data["data.py<br/>citire/scriere etichete"]
    end

    subgraph ML["src/pi_p (pipeline ML, linie de comandă)"]
        prepare["prepare.py"]
        check["check.py"]
        eda["eda.py"]
        tile["tile.py"]
        train["train.py"]
        evaluate["evaluate.py"]
        config["config.py"]
    end

    YOLO[["Ultralytics YOLO26"]]

    main --> dashboard & library_ts & auth_ts & api_ts & dom & types
    dashboard --> charts & dom
    library_ts --> auth_ts & dom
    api_ts --> auth_ts

    main -. "WebSocket /api/stream" .-> server
    api_ts -. "HTTP /api/*" .-> server
    library_ts -. "HTTP /api/library" .-> server
    auth_ts -. "HTTP /api/login" .-> server

    server --> session & library_py & auth_py
    session --> analytics & scoremap & predict & render & slicing & YOLO
    render --> analytics & data
    predict --> slicing & YOLO

    prepare & check & eda & tile --> data
    tile --> slicing
    train --> config & YOLO
    evaluate --> YOLO
```

## 3. Diagrama de clase

Python nu obligă la clase, așa că modulele formate doar din funcții apar cu stereotipul `«module»`.

```mermaid
classDiagram
    direction LR

    class Auth {
        -totp: TOTP
        -key: bytes
        -attempts: dict~str, deque~
        -last_step: int
        +enabled: bool
        +allow_attempt(client) bool
        +verify(code) bool
        +issue() str
        +valid(token) bool
        -_sign(expires) str
    }

    class Analytics {
        +names: dict
        +history: deque~Counter~
        +tracks: dict~int, deque~
        +seen: set~int~
        +flow: list
        +dwell: ndarray
        +decay: float
        +update(xyxy, cls, ids, inference_ms)
        +occupancy() ndarray
        +peak(occ)$ tuple
        +summary() dict
        -_accumulate(xyxy)
        -_update_tracks(xyxy, cls, ids)
    }

    class ScoreMap {
        +logits: dict
        +input_hw: tuple
        +heatmap(h, w) ndarray
        -_capture_input(module, args)
        -_capture(branch, i)
    }

    class Params {
        <<dataclass>>
        +conf: float = 0.25
        +view: str = "detections"
    }

    class Session {
        +cap: VideoCapture
        +model: YOLO
        +scores: ScoreMap
        +analytics: Analytics
        +params: Params
        +trails: dict
        +writer: VideoWriter
        +rows: list~dict~
        +last: tuple
        +meta(weights) dict
        +detect(frame) tuple
        +step() tuple
        +render(cam) ndarray
        +redraw() bytes
        +close() list~str~
    }

    class server {
        <<module>>
        +auth: Auth
        +slots: Semaphore
        +require_login(request, call_next)
        +login(request, response)
        +upload(file)
        +library()
        +stream(ws)
        +run_session(ws, video, out_dir, ...)
        +parse_start(msg) tuple
        +cleanup()
    }

    class render {
        <<module>>
        +draw_frame(img, xyxy, conf, cls, ids, trails, names)
        +heat_overlay(img, cam)
        +congestion_overlay(img, occupancy, peak)
        +draw_grid(img, wins)
        +resize_to(img, width)
    }

    class library {
        <<module>>
        +list_videos(root) list
        +resolve(root, name) Path
        +thumbnail(path, mtime) bytes
        +probe(path, mtime) dict
    }

    class predict {
        <<module>>
        +sliced_predict(model, img, size, overlap, conf, iou, device)
        +drop_cut(det, win, w, h)
        +merge(det, thr)
    }

    class slicing {
        <<module>>
        +windows(w, h, size, overlap) list
        +crop_boxes(xyxy, win, min_visibility)
    }

    class YOLO {
        <<extern: Ultralytics>>
        +predict()
        +track()
        +train()
        +val()
    }

    Session *-- Params
    Session *-- Analytics
    Session *-- ScoreMap
    Session o-- YOLO
    ScoreMap ..> YOLO : hook-uri pe capul de detecție
    server --> Auth
    server ..> Session : creează câte una pe sesiune
    server ..> library
    Session ..> render
    Session ..> predict
    predict ..> slicing
    render ..> Analytics : JAM_LEVEL
```

## 4. Diagrama de secvență: sesiune de analiză

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilizator
    participant B as Browser (main.ts)
    participant S as server.py
    participant SE as Session
    participant M as YOLO (GPU)

    U->>B: introduce codul din Google Authenticator
    B->>S: POST /api/login {code}
    S->>S: Auth.allow_attempt() și Auth.verify()
    S-->>B: 200 + cookie pi_p_session (HttpOnly)

    alt video din bibliotecă
        U->>B: Bibliotecă → Analizează
    else upload (sub 100 MB)
        U->>B: alege fișier
        B->>S: POST /api/videos
        S-->>B: {id}
    end

    U->>B: Pornește
    B->>S: WebSocket /api/stream (cu cookie)
    S->>S: Auth.valid(cookie)
    B->>S: {"action":"start", sursa, model, prag, ...}
    S->>S: parse_start() validează tot
    S->>SE: Session(video, model, ...)
    SE->>M: încarcă modelul și face încălzirea
    S-->>B: {"type":"meta", clase, culori, fps, ...}

    loop pentru fiecare cadru
        S->>SE: step()
        SE->>M: detect(frame)
        M-->>SE: cutii, încredere, clase, ID-uri
        SE-->>S: imagine JPEG, hartă congestie, statistici
        S-->>B: binar 0x01 + cadrul adnotat
        S-->>B: binar 0x02 + harta de congestie (1/s)
        S-->>B: {"type":"metrics", ...}
        B->>B: dashboard.ts desenează panourile
    end

    opt pauză
        U->>B: click pe video / Space
        B->>S: {"action":"pause"}
        Note over S: bucla așteaptă, nu procesează nimic
        U->>B: schimbă pragul sau vizualizarea
        B->>S: {"action":"params", conf, view}
        S->>SE: redraw()
        S-->>B: cadrul curent redesenat
        B->>S: {"action":"resume"}
    end

    alt utilizatorul apasă Oprește
        B->>S: {"action":"stop"}
    else video terminat
        SE-->>S: step() întoarce None
    end
    S->>SE: close() scrie detectii.mp4 și numaratoare.csv
    S-->>B: {"type":"done", files}
    U->>B: descarcă rezultatele
```

## 5. Diagrama de secvență: procesarea unui cadru

Ce se întâmplă în `Session.step()` pentru un singur cadru.

```mermaid
sequenceDiagram
    autonumber
    participant SE as Session.step()
    participant CV as OpenCV
    participant L as Lacăt GPU
    participant M as YOLO
    participant SM as ScoreMap
    participant A as Analytics
    participant R as render.py

    SE->>CV: cap.read()
    CV-->>SE: cadru BGR (ex. 1920×1080)
    SE->>CV: resize_to(cadru, 1280)
    SE->>L: acquire (un singur fir pe GPU)
    alt inferență pe bucăți
        SE->>M: sliced_predict(): imaginea întreagă + ferestre suprapuse
        Note over M: drop_cut() elimină obiectele tăiate de margini<br/>merge() combină duplicatele după IOS
    else urmărire
        SE->>M: model.track(persist=True) cu ByteTrack
    end
    M-->>SE: xyxy, conf, cls, ids
    opt vizualizare = hartă de scoruri
        SE->>SM: heatmap(h, w)
        SM-->>SE: probabilitatea maximă pe clase, per pixel
    end
    SE->>L: release
    SE->>SE: păstrează detecțiile cu conf ≥ prag
    SE->>A: update(cutii, clase, ID-uri, ms)
    Note over A: istoric pe clase, traiectorii,<br/>hartă de staționare cu scădere exponențială
    SE->>R: draw_frame() cutii, etichete, traiectorii
    SE->>CV: VideoWriter.write() în detectii.mp4
    opt a trecut 1 s de la ultima hartă
        SE->>A: occupancy() și peak()
        SE->>R: congestion_overlay()
    end
    SE->>A: summary()
    SE-->>SE: întoarce (JPEG, JPEG congestie sau None, statistici)
```

## 6. Diagrama de activitate: pipeline-ul de date și antrenare

```mermaid
flowchart TD
    start((start)) --> prep["pi-prepare<br/>descarcă VisDrone, convertește în format YOLO"]
    prep --> mask{"zone «ignored» sau clasa «others»?"}
    mask -- da --> fill["le acoperă cu gri 114<br/>păstrând obiectele etichetate"]
    mask -- nu --> labels
    fill --> labels["scrie etichetele normalizate<br/>clasă, x, y, w, h"]
    labels --> check["pi-check<br/>imagini corupte, etichete invalide, duplicate"]
    check --> ok{"erori?"}
    ok -- da --> fix["corectează datele"] --> check
    ok -- nu --> eda["pi-eda<br/>clase, mărimi, densitate, heatmap"]
    eda --> choice{"strategie"}
    choice -- "640 px" --> t640["pi-train configs/train.yaml"]
    choice -- "1024 px" --> t1024["pi-train imgsz=1024"]
    choice -- "tile-uri" --> tile["pi-tile<br/>ferestre 640 px, suprapunere 20%"] --> ttile["pi-train configs/train_tiled.yaml"]
    t640 & t1024 & ttile --> eval["pi-eval<br/>mAP pe clase"]
    eval --> best["alege cel mai bun model<br/>runs/*/weights/best.pt"]
    best --> app["dashboard: modelul apare automat în listă"]
    app --> stop((sfârșit))
```

## 7. Diagrama de stări: interfața

```mermaid
stateDiagram-v2
    [*] --> Verificare: deschide pagina
    Verificare --> Login: autentificare activă, fără sesiune
    Verificare --> Pregătit: sesiune validă
    Login --> Pregătit: cod corect
    Login --> Login: cod greșit / prea multe încercări

    state Pregătit {
        [*] --> FărăSursă
        FărăSursă --> Încărcare: alege fișier
        Încărcare --> SursăAleasă: upload reușit
        Încărcare --> FărăSursă: eroare / peste 100 MB
        FărăSursă --> SursăAleasă: alege din Bibliotecă
    }

    Pregătit --> Conectare: Pornește
    Conectare --> Rulare: primește meta
    Conectare --> Pregătit: serverul refuză
    Rulare --> Pauză: click / Space
    Pauză --> Rulare: click / Space
    Rulare --> Terminat: video terminat / Oprește
    Pauză --> Terminat: Oprește
    Terminat --> Pregătit: rezultate disponibile
    Rulare --> Login: sesiune expirată (4401)
    Pregătit --> Login: Ieșire
```
