# Rezultate

Toate valorile de mai jos sunt măsurate cu `pi-eval` pe cele 548 de imagini întregi din **val**, cu prag de încredere 0,01, pe un RTX 4070 SUPER. Cum se calculează fiecare coloană este explicat în [pipeline/08-evaluarea.md](pipeline/08-evaluarea.md). Subsetul test-dev nu a fost rulat încă: el se rulează o singură dată, după ce toate deciziile sunt luate.

Rezultatele brute sunt în `logs/eval/*.json` (folder neinclus în Git).

## Starea experimentelor

| Etapa din [plan](benchmarkuri.md) | Stare |
|---|---|
| 1. Împărțirea imaginii | **făcută**; SAHI ca bibliotecă nu a fost rulat, „baseline + grilă” ține locul lui |
| 2. CLAHE | cod gata (`pi-stats`, `pi-clahe`); **reantrenările nu au fost rulate** |
| 3. Ferestre adaptive | cod gata; măsurat pe modelul din etapa 1, nu pe configurația din etapa 2 |
| Test final pe test-dev | nefăcut |

## Antrenările

| Model | Configurare | Date de antrenare | Epoci | Durată | Cea mai bună epocă |
|---|---|---|---|---|---|
| baseline | `configs/train.yaml` | 6471 imagini întregi | 97 (oprit de `patience`) | 1,2 ore | 77 |
| pe felii | `configs/train_tiled.yaml` | 46 483 felii și imagini întregi | 100 | 7,7 ore | 82 |

## Etapa 1: cum împărțim imaginea

| Variantă | Antrenare | Testare (`--mode`) | mAP50 | mAP50-95 | Precizie | Recall | ms/imagine |
|---|---|---|---|---|---|---|---|
| Baseline | imagini întregi | `intreg` | 0,307 | 0,173 | 0,481 | 0,336 | 10,5 |
| Baseline + grilă | imagini întregi | `grila` | 0,395 | 0,224 | 0,489 | 0,451 | 44,3 |
| Model pe felii, imagine întreagă | felii | `intreg` | 0,326 | 0,189 | 0,540 | 0,346 | 7,7 |
| **Model pe felii + grilă** | felii | `grila` | **0,483** | **0,292** | 0,610 | 0,506 | 35,1 |

Ce arată tabelul:

- **Felierea la testare ajută și fără reantrenare:** +0,088 mAP50 (rândul 2 față de rândul 1).
- **Antrenarea pe felii adaugă încă +0,088** peste asta (rândul 4 față de rândul 2).
- **Antrenarea pe felii, singură, ajută puțin:** +0,019 (rândul 3 față de rândul 1). Câștigul vine din combinația „antrenat pe felii, testat pe felii”.
- **Costul este timpul:** de aproximativ 4 ori mai mult pe imagine. Imaginile din val au în medie 5,2 ferestre de grilă.

Configurația selectată pentru etapele următoare: **model pe felii + grilă**.

Timpii sunt orientativi: două rulări ale aceleiași rețele pe imaginea întreagă au dat 10,5 și 7,7 ms, deci diferențele sub ~30% nu înseamnă nimic.

### Pe clase (mAP50)

| Clasă | Baseline | Model pe felii + grilă | Diferență |
|---|---|---|---|
| pedestrian | 0,370 | 0,606 | +0,236 |
| people | 0,260 | 0,449 | +0,189 |
| bicycle | 0,084 | 0,285 | +0,201 |
| car | 0,752 | 0,839 | +0,087 |
| van | 0,281 | 0,437 | +0,156 |
| truck | 0,264 | 0,434 | +0,170 |
| tricycle | 0,203 | 0,369 | +0,166 |
| awning-tricycle | 0,072 | 0,163 | +0,091 |
| bus | 0,393 | 0,632 | +0,239 |
| motor | 0,395 | 0,618 | +0,223 |

Toate clasele cresc. Mașinile, deja bine detectate, câștigă cel mai puțin. Cel mai mult câștigă pedestrian, bus, motor și bicycle (care se triplează). Creșterea la clasele cu obiecte mici este în acord cu ipoteza de la care a pornit felierea; nu am măsurat însă separat pe mărimi de obiecte, deci nu putem spune cât din câștig vine de la ele.

## Grupele de iluminare

Praguri calculate de `pi-stats` pe train (percentila 15): luminozitate < 61,6 înseamnă „întuneric”, contrast < 36,0 înseamnă „contrast slab”.

| Subset | întuneric | contrast slab | normal |
|---|---|---|---|
| train | 971 | 556 | 4944 |
| val | 9 | 89 | 450 |
| test | 444 | 67 | 1099 |

**Val are doar 9 imagini întunecate.** Un mAP calculat pe 9 imagini se schimbă mult de la o imagine la alta, deci coloana „întuneric” de mai jos nu poate susține nicio concluzie. Pentru etapa 2 (CLAHE), asta e o problemă reală: efectul pe imaginile de noapte se va vedea bine abia pe test, care are 444.

mAP50 pe grupe, pe val:

| Variantă | întuneric (9) | contrast slab (89) | normal (450) |
|---|---|---|---|
| Baseline | 0,377 | 0,327 | 0,302 |
| Model pe felii + grilă | 0,574 | 0,519 | 0,480 |

Imaginile cu contrast slab nu ies mai prost decât cele normale, ba chiar puțin mai bine. Asta nu înseamnă că iluminarea slabă ajută: grupele diferă și prin scenă și prin numărul de obiecte.

## Etapa 3 (preliminar): ferestre adaptive

Măsurat pe modelul pe felii, fără CLAHE și fără reantrenare.

Au fost scrise două variante de ferestre adaptive, independent. Ambele au fost măsurate cu `pi-eval`, în aceleași condiții:

- **A**, ferestre fixe de 640 px puse ca o mică grilă peste fiecare grup, la rezoluția reală;
- **B**, câte o fereastră pe grup, de mărime variabilă (cel puțin 640 px), redimensionată la 640. Este varianta păstrată în cod (`src/pi_p/adaptive.py`, explicată în [etapa 9](pipeline/09-ferestre-adaptive.md)).

| Testare | mAP50 | mAP50-95 | Recall | Ferestre pe imagine | Suprafața ferestrelor | ms/imagine* |
|---|---|---|---|---|---|---|
| `grila` | 0,483 | 0,292 | 0,506 | 5,2 | – | 35,1 |
| adaptiv A (scos din cod) | 0,490 | 0,296 | 0,515 | 5,5 | 2,3 × imaginea | 102 |
| **`adaptiv` (B)** | **0,502** | **0,304** | 0,521 | 10,2 | 4,5 × imaginea | 212 |
| `adaptiv+grila` (B + grilă) | 0,507 | 0,307 | 0,526 | – | – | 172 |

\* Coloanele A și B au fost măsurate în timp ce placa video rula o antrenare, deci timpii absoluți sunt mult umflați. Comparabil este doar raportul dintre A și B, măsurate unul după altul: B e de aproximativ 2 ori mai lentă. `grila` a fost măsurată cu placa liberă.

Ce arată tabelul:

- **Varianta B detectează cel mai bine:** +0,019 mAP50 față de grilă și +0,012 față de A. Diferențele sunt pe o singură rulare; nu am verificat dacă sunt peste zgomotul măsurătorii.
- **Câștigul vine din mai multă muncă, nu din ferestre mai bine alese.** B rulează modelul pe aproape de 2 ori mai multe ferestre decât grila, care se suprapun mult (4,5 × suprafața imaginii). Fiecare grup mic primește o fereastră întreagă de 640 × 640.
- **Niciuna dintre variante nu economisește timp pe val**, iar acesta era scopul ferestrelor adaptive. Scenele din VisDrone sunt aglomerate, iar obiectele sunt răspândite în tot cadrul.
- **Parametrii nu au fost reglați.** Valorile din `configs/adaptive.yaml` sunt cele de pornire. Un nucleu de dilatare mai mare ar uni grupurile mici și ar da mai puține ferestre.

Concluzia de moment: ferestrele adaptive îmbunătățesc puțin detecția, dar nu își ating scopul (viteza). Următorul pas este reglarea parametrilor cu `pi-adaptive-eval`, de exemplu `dilate_kernel` și `prelim_conf`, urmărind numărul de ferestre.

## Cum reproduci tabelele

```bash
B=runs/detect/runs/yolo26n_640/weights/best.pt
T=runs/detect/runs/yolo26n_tiled/weights/best.pt

uv run pi-stats
uv run pi-eval $B --mode intreg        --out logs/eval/baseline_intreg.json
uv run pi-eval $B --mode grila         --out logs/eval/baseline_grila.json
uv run pi-eval $T --mode intreg        --out logs/eval/tiled_intreg.json
uv run pi-eval $T --mode grila         --out logs/eval/tiled_grila.json
uv run pi-eval $T --mode adaptiv       --out logs/eval/tiled_adaptiv.json
uv run pi-eval $T --mode adaptiv+grila --out logs/eval/tiled_adaptiv_grila.json
```
