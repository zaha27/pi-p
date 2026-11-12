# Arhitectura proiectului (UML)

Diagramele sunt scrise in Mermaid si se afiseaza direct pe GitHub.

1. [Deployment](#1-diagrama-de-deployment): unde ruleaza fiecare parte
2. [Componente](#2-diagrama-de-componente): modulele si dependentele dintre ele
3. [Clase](#3-diagrama-de-clase): clasele si functiile principale
4. [Secventa: sesiune de analiza](#4-diagrama-de-secventa-sesiune-de-analiza)
5. [Secventa: procesarea unui cadru](#5-diagrama-de-secventa-procesarea-unui-cadru)
6. [Activitate: pipeline-ul de date si antrenare](#6-diagrama-de-activitate-pipeline-ul-de-date-si-antrenare)
7. [Stari: interfata](#7-diagrama-de-stari-interfata)

## 1. Diagrama de deployment

```mermaid
flowchart LR
    subgraph Client["Dispozitivul utilizatorului"]
        B["Browser<br/>dashboard Vite + TypeScript"]
        GA["Google Authenticator<br/>genereaza coduri TOTP"]
    end

    subgraph CF["Cloudflare"]
        DNS["DNS<br/>pi.zaha.cc → tunel"]
        EDGE["Edge<br/>HTTPS, limita 100 MB/cerere"]
    end

    subgraph Host["Server: PC cu RTX 4070 Super sau Mac"]
        subgraph Docker["docker compose"]
            T["container tunnel<br/>cloudflared"]
            A["container app<br/>FastAPI + Uvicorn :8000"]
        end
        V[("videos/<br/>biblioteca")]
        R[("runs/<br/>modele antrenate")]
        W[("/tmp/pi-p<br/>upload-uri si rezultate")]
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

Containerul `app` nu e expus direct in retea: portul 8000 e legat doar de `127.0.0.1`. Din internet se ajunge numai prin tunel, iar orice cerere `/api/*` cere autentificare.

## 2. Diagrama de componente

```mermaid
flowchart TB
    subgraph FE["infrastructure/frontend/src (browser)"]
        main["main.ts<br/>controale, WebSocket, pauza"]
        dashboard["dashboard.ts<br/>deseneaza panourile"]
        charts["charts.ts<br/>sparkline, flux, camp de miscare"]
        library_ts["library.ts<br/>tabul Biblioteca"]
        auth_ts["auth.ts<br/>ecranul de login"]
        api_ts["api.ts<br/>upload, modele"]
        dom["dom.ts"]
        types["types.ts<br/>mesajele serverului"]
    end

    subgraph APP["infrastructure/pi_p_server (server web)"]
        server["server.py<br/>rute HTTP, WebSocket, login"]
        session["session.py<br/>Session: un video procesat"]
        render["render.py<br/>desen cutii, harti"]
        library_py["library.py<br/>listare, thumbnail"]
        auth_py["auth.py<br/>TOTP, sesiuni semnate"]
    end

    subgraph CORE["src/pi_p (logica de baza)"]
        analytics["analytics.py<br/>flux, congestie, statistici"]
        scoremap["scoremap.py<br/>harta de scoruri a detectorului"]
        predict["predict.py<br/>grila: drop_cut, to_image, merge"]
        adaptive["adaptive.py<br/>ferestre adaptive"]
        slicing["slicing.py<br/>ferestre suprapuse"]
        data["data.py<br/>citire/scriere etichete"]
    end

    subgraph ML["src/pi_p (pipeline ML, linie de comanda)"]
        prepare["prepare.py"]
        check["check.py"]
        eda["eda.py"]
        stats["stats.py<br/>grupe de iluminare"]
        enhance["enhance.py<br/>CLAHE"]
        tile["tile.py"]
        train["train.py"]
        evaluate["evaluate.py"]
        metrics["metrics.py<br/>potrivire, mAP"]
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
    adaptive --> predict & config
    adaptive_figure["adaptive_figure.py<br/>figuri"] --> adaptive

    prepare & check & eda & tile & stats & enhance --> data
    enhance --> stats
    tile --> slicing
    train --> config & YOLO
    evaluate --> predict & adaptive & metrics & stats & YOLO
```

## 3. Diagrama de clase

Python nu obliga la clase, asa ca modulele formate doar din functii apar cu stereotipul `«module»`.

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
        +to_image(det, win, w, h)
        +merge(det, thr)
    }

    class adaptive {
        <<module>>
        +load_params(path, overrides) AdaptiveParams
        +adaptive_trace(model, img, params, device) AdaptiveTrace
        +adaptive_predict(model, img, params, device) (det, windows)
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
    ScoreMap ..> YOLO : hook-uri pe capul de detectie
    server --> Auth
    server ..> Session : creeaza cate una pe sesiune
    server ..> library
    Session ..> render
    Session ..> predict
    predict ..> slicing
    adaptive ..> predict
    render ..> Analytics : JAM_LEVEL
```

## 4. Diagrama de secventa: sesiune de analiza

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
    S->>S: Auth.allow_attempt() si Auth.verify()
    S-->>B: 200 + cookie pi_p_session (HttpOnly)

    alt video din biblioteca
        U->>B: Biblioteca → Analizeaza
    else upload (sub 100 MB)
        U->>B: alege fisier
        B->>S: POST /api/videos
        S-->>B: {id}
    end

    U->>B: Porneste
    B->>S: WebSocket /api/stream (cu cookie)
    S->>S: Auth.valid(cookie)
    B->>S: {"action":"start", sursa, model, prag, ...}
    S->>S: parse_start() valideaza tot
    S->>SE: Session(video, model, ...)
    SE->>M: incarca modelul si face incalzirea
    S-->>B: {"type":"meta", clase, culori, fps, ...}

    loop pentru fiecare cadru
        S->>SE: step()
        SE->>M: detect(frame)
        M-->>SE: cutii, incredere, clase, ID-uri
        SE-->>S: imagine JPEG, harta congestie, statistici
        S-->>B: binar 0x01 + cadrul adnotat
        S-->>B: binar 0x02 + harta de congestie (1/s)
        S-->>B: {"type":"metrics", ...}
        B->>B: dashboard.ts deseneaza panourile
    end

    opt pauza
        U->>B: click pe video / Space
        B->>S: {"action":"pause"}
        Note over S: bucla asteapta, nu proceseaza nimic
        U->>B: schimba pragul sau vizualizarea
        B->>S: {"action":"params", conf, view}
        S->>SE: redraw()
        S-->>B: cadrul curent redesenat
        B->>S: {"action":"resume"}
    end

    alt utilizatorul apasa Opreste
        B->>S: {"action":"stop"}
    else video terminat
        SE-->>S: step() intoarce None
    end
    S->>SE: close() scrie detectii.mp4 si numaratoare.csv
    S-->>B: {"type":"done", files}
    U->>B: descarca rezultatele
```

## 5. Diagrama de secventa: procesarea unui cadru

Ce se intampla in `Session.step()` pentru un singur cadru.

```mermaid
sequenceDiagram
    autonumber
    participant SE as Session.step()
    participant CV as OpenCV
    participant L as Lacat GPU
    participant M as YOLO
    participant SM as ScoreMap
    participant A as Analytics
    participant R as render.py

    SE->>CV: cap.read()
    CV-->>SE: cadru BGR (ex. 1920×1080)
    SE->>CV: resize_to(cadru, 1280)
    SE->>L: acquire (un singur fir pe GPU)
    alt inferenta pe bucati
        SE->>M: sliced_predict(): imaginea intreaga + ferestre suprapuse
        Note over M: drop_cut() elimina obiectele taiate de margini<br/>merge() combina duplicatele dupa IOS
    else urmarire
        SE->>M: model.track(persist=True) cu ByteTrack
    end
    M-->>SE: xyxy, conf, cls, ids
    opt vizualizare = harta de scoruri
        SE->>SM: heatmap(h, w)
        SM-->>SE: probabilitatea maxima pe clase, per pixel
    end
    SE->>L: release
    SE->>SE: pastreaza detectiile cu conf ≥ prag
    SE->>A: update(cutii, clase, ID-uri, ms)
    Note over A: istoric pe clase, traiectorii,<br/>harta de stationare cu scadere exponentiala
    SE->>R: draw_frame() cutii, etichete, traiectorii
    SE->>CV: VideoWriter.write() in detectii.mp4
    opt a trecut 1 s de la ultima harta
        SE->>A: occupancy() si peak()
        SE->>R: congestion_overlay()
    end
    SE->>A: summary()
    SE-->>SE: intoarce (JPEG, JPEG congestie sau None, statistici)
```

## 6. Diagrama de activitate: pipeline-ul de date si antrenare

```mermaid
flowchart TD
    start((start)) --> prep["pi-prepare<br/>descarca VisDrone, converteste in format YOLO"]
    prep --> mask{"zone «ignored» sau clasa «others»?"}
    mask -- da --> fill["le acopera cu gri 114<br/>pastrand obiectele etichetate"]
    mask -- nu --> labels
    fill --> labels["scrie etichetele normalizate<br/>clasa, x, y, w, h"]
    labels --> check["pi-check<br/>imagini corupte, etichete invalide, duplicate"]
    check --> ok{"erori?"}
    ok -- da --> fix["corecteaza datele"] --> check
    ok -- nu --> eda["pi-eda<br/>clase, marimi, densitate, heatmap"]
    eda --> stats["pi-stats<br/>luminozitate, contrast, grupe"]
    stats --> choice{"strategie"}
    choice -- "640 px" --> t640["pi-train configs/train.yaml"]
    choice -- "1024 px" --> t1024["pi-train imgsz=1024"]
    choice -- "tile-uri" --> tile["pi-tile<br/>ferestre 640 px, suprapunere 20%"] --> ttile["pi-train configs/train_tiled.yaml"]
    choice -- "CLAHE" --> clahe["pi-clahe --mode uniform / adaptive"] --> tile
    t640 & t1024 & ttile --> eval["pi-eval<br/>mAP pe imagini intregi, pe clase si pe grupe"]
    eval --> best["alege cel mai bun model<br/>runs/*/weights/best.pt"]
    best --> app["dashboard: modelul apare automat in lista"]
    app --> stop((sfarsit))
```

## 7. Diagrama de stari: interfata

```mermaid
stateDiagram-v2
    [*] --> Verificare: deschide pagina
    Verificare --> Login: autentificare activa, fara sesiune
    Verificare --> Pregatit: sesiune valida
    Login --> Pregatit: cod corect
    Login --> Login: cod gresit / prea multe incercari

    state Pregatit {
        [*] --> FaraSursa
        FaraSursa --> Incarcare: alege fisier
        Incarcare --> SursaAleasa: upload reusit
        Incarcare --> FaraSursa: eroare / peste 100 MB
        FaraSursa --> SursaAleasa: alege din Biblioteca
    }

    Pregatit --> Conectare: Porneste
    Conectare --> Rulare: primeste meta
    Conectare --> Pregatit: serverul refuza
    Rulare --> Pauza: click / Space
    Pauza --> Rulare: click / Space
    Rulare --> Terminat: video terminat / Opreste
    Pauza --> Terminat: Opreste
    Terminat --> Pregatit: rezultate disponibile
    Rulare --> Login: sesiune expirata (4401)
    Pregatit --> Login: Iesire
```
