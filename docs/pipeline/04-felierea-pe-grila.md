# 4. Felierea pe grilă fixă

**Cod:** `src/pi_p/slicing.py` (grila), `src/pi_p/tile.py` (felii pentru antrenare) · **Comandă:** `pi-tile`

## De ce e nevoie de etapa asta

YOLO primește imagini de 640 px pe latura mare. O imagine VisDrone de 1920 × 1080 e micșorată de 3 ori, deci un pieton de 12 × 30 px ajunge la 4 × 10 px, prea mic ca rețeaua să îl mai distingă.

Soluția: în loc să micșorăm imaginea, o tăiem în ferestre de 640 × 640 și dăm modelului fiecare fereastră la rezoluția ei reală. Pietonul rămâne de 12 × 30 px.

## Cum se așază ferestrele

Ferestrele se suprapun cu 20%, adică pasul dintre ele este 640 × (1 − 0.2) = 512 px. Ultima fereastră de pe fiecare rând și coloană e lipită de marginea imaginii, ca să nu rămână nimic neacoperit.

Exemplu pe o imagine de 1360 × 765:

```
lățime 1360:  ferestre care încep la x = 0, 512, 720      (720 = 1360 − 640)
înălțime 765: ferestre care încep la y = 0, 125           (125 = 765 − 640)

pe orizontală, fiecare fereastră acoperă 640 px:

    x:  0        512  640 720        1152      1360
    A   ├──────────────┤
    B            ├──────────────────────┤
    C                      ├──────────────────────┤

3 coloane × 2 rânduri = 6 ferestre
```

Ferestrele A și B au în comun banda 512–640. Ultima fereastră, C, nu pornește la 1024 (ar ieși din imagine), ci e trasă înapoi la 720, deci se suprapune mai mult cu B.

Funcția `starts` calculează pozițiile pe o latură, iar `windows` le combină pe cele două laturi.

## De ce se suprapun ferestrele

Un obiect aflat exact pe granița dintre două ferestre ar fi tăiat în două și nu ar apărea întreg nicăieri. Cu o suprapunere de 128 px, orice obiect mai mic de 128 px apare întreg în cel puțin o fereastră. Prețul sunt detecțiile duble din zona comună, rezolvate la [fuziune](07-fuziunea-detectiilor.md).

## Felierea pentru antrenare (`pi-tile`)

Ca modelul să fie antrenat pe același tip de imagine pe care îl primește la test, tăiem și imaginile de antrenare în felii.

```mermaid
flowchart TB
    A["Imagine + etichete"] --> F["copie a imaginii întregi
(obiectele mari)"]
    A --> W["windows
grila de ferestre"]
    W --> C["crop_boxes
mută etichetele în fereastră"]
    C --> V{"fereastra are obiecte?"}
    V -- "da" --> S["scrie felia + etichetele"]
    V -- "nu" --> E{"noroc < 10%?"}
    E -- "da" --> S
    E -- "nu" --> X["aruncată"]
    F --> O[("datasets/VisDrone-tiled")]
    S --> O
```

Trei decizii, fiecare cu motivul ei:

| Decizie | Parametru | De ce |
|---|---|---|
| Un obiect tăiat de fereastră e păstrat doar dacă rămâne vizibil cel puțin 50% din el | `--min-visibility 0.5` | altfel modelul învață să detecteze „un colț de mașină” |
| Feliile fără obiecte sunt păstrate doar în proporție de 10% | `--empty 0.1` | modelul trebuie să vadă și fundal, dar nu să fie dominat de el |
| Se păstrează și imaginea întreagă | implicit; `--no-full` o scoate | obiectele mari (autobuze aproape de cameră) nu încap într-o felie |

Etichetele se mută în coordonatele feliei scăzând colțul ei stânga-sus (`crop_boxes`). Numele feliei spune de unde vine: `0000001_512_125.jpg` este felia care începe la x = 512, y = 125.

## Rezultatul pe VisDrone

| Subset | Imagini | Fișiere după feliere |
|---|---|---|
| train | 6471 | 46 483 |
| val | 548 | 3 320 |

Setul de antrenare crește de aproximativ 7 ori, deci și timpul unei epoci: de la ~45 de secunde la ~5 minute pe RTX 4070 SUPER.

## Comenzi

```bash
uv run pi-tile                       # datasets/VisDrone → datasets/VisDrone-tiled
uv run pi-check --root datasets/VisDrone-tiled
```

Feliile din `val` sunt folosite doar de Ultralytics, ca să urmărească antrenarea. Rezultatele raportate se calculează altfel, pe imaginile întregi ([evaluarea](08-evaluarea.md)).
