# Codul, explicat

Ghid pentru prezentare: ideile importante din cod, în ordinea pipeline-ului, fiecare cu fragmentul care o implementează și un exemplu. Legătura dintre comenzi e în [pipeline.md](pipeline.md).

## 1. Datele

Etichetele sunt în formatul YOLO: un fișier `.txt` pentru fiecare imagine, cu câte un rând `clasă x_centru y_centru lățime înălțime` pentru fiecare obiect. Coordonatele sunt împărțite la dimensiunile imaginii, deci sunt între 0 și 1. Funcțiile de citire și scriere sunt în [data.py](../src/pi_p/data.py), iar restul modulelor le folosesc doar pe acestea.

VisDrone marchează unele zone ca „ignored” (mulțimi pe care adnotatorii nu le-au etichetat) și are o clasă „others” pe care nu o folosim. Dacă rămân în imagine, modelul e penalizat pentru obiecte reale pe care le găsește acolo. `pi-prepare` le acoperă cu gri, dar lasă neatinse obiectele etichetate care se suprapun cu ele ([prepare.py](../src/pi_p/prepare.py)):

```python
def mask_ignored(img, keep, ignore):
    m = np.zeros(img.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in ignore:
        m[y1:y2, x1:x2] = True
    for x1, y1, x2, y2 in keep:
        m[y1:y2, x1:x2] = False
    img[m] = FILL  # 114, griul folosit și de YOLO la completarea imaginilor
```

## 2. Analiza statistică

Pentru fiecare imagine se calculează media și abaterea standard a canalului L din spațiul de culoare LAB, adică luminozitatea și contrastul ([stats.py](../src/pi_p/stats.py)). Imaginea e citită micșorată de 4 ori, pentru că media și abaterea abia se schimbă, iar citirea e mult mai rapidă.

```python
def group(brightness, contrast, dark=70, low_contrast=35):
    if brightness < dark:
        return "dark"
    if contrast < low_contrast:
        return "low_contrast"
    return "normal"
```

Pragurile 70 și 35 se pot schimba cu `--dark` și `--low-contrast`, iar histogramele din `stats.png` arată unde cad față de distribuție. Rezultatul pe VisDrone:

| Split | normal | dark | low_contrast |
|---|---|---|---|
| train | 4736 | 1139 | 596 |
| val | 431 | 13 | 104 |
| test-dev | 1032 | 501 | 77 |

Pe val sunt doar 13 imagini întunecate, așa că mAP-ul pe grupa `dark` din val e foarte zgomotos. Concluziile despre imaginile întunecate trebuie verificate pe test-dev, unde sunt 501.

## 3. CLAHE

CLAHE egalizează histograma pe bucăți de imagine (grilă de 8×8) și limitează cât poate crește contrastul (`clip=2.0`), ca să nu amplifice zgomotul. Se aplică doar pe canalul L, deci culorile rămân aceleași ([enhance.py](../src/pi_p/enhance.py)):

```python
def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def needs_clahe(name, mode, groups):
    return mode == "uniform" or groups.get(name) in {"dark", "low_contrast"}
```

Varianta `uniform` aplică CLAHE pe toate imaginile, iar `adaptive` doar pe cele din grupele slabe. Etichetele se copiază neschimbate, pentru că CLAHE nu mută pixelii.

## 4. Grila de ferestre

Ferestrele au 640 px și se suprapun 20%, deci pasul e 512 px. Ultima fereastră de pe fiecare axă e lipită de marginea imaginii, ca să nu rămână o bandă neacoperită ([slicing.py](../src/pi_p/slicing.py)):

```python
def starts(length, size, step):
    if length <= size:
        return [0]
    return [*range(0, length - size, step), length - size]
```

Exemplu: o imagine VisDrone de 1360×765 primește ferestrele care încep la x ∈ {0, 512, 720} și y ∈ {0, 125}, adică 6 ferestre.

La antrenarea pe felii, `pi-tile` decupează aceleași ferestre și păstrează un obiect doar dacă cel puțin jumătate din el se vede în fereastră (`--min-visibility 0.5`). Altfel, modelul ar învăța să recunoască bucăți mici de mașini ca mașini întregi. Ferestrele fără obiecte se păstrează doar în proporție de 10% (`--empty 0.1`), ca modelul să vadă și fundal, dar fără să fie copleșit de el.

## 5. Fuziunea detecțiilor

Detecțiile dintr-o fereastră sunt în coordonatele ferestrei. Înainte de unire se fac doi pași ([predict.py](../src/pi_p/predict.py)):

```python
def to_image(det, win, w, h):
    return drop_cut(det, win, w, h) + torch.tensor([win[0], win[1], win[0], win[1], 0, 0])
```

1. `drop_cut` aruncă detecțiile care ating o margine a ferestrei aflată în interiorul imaginii: obiectul e tăiat acolo și apare întreg în fereastra vecină. O detecție lipită de marginea reală a imaginii rămâne. De exemplu, în fereastra care începe la x = 512, o casetă cu x1 = 0 e aruncată.
2. Se adună colțul stânga-sus al ferestrei, ca să ajungă în coordonatele imaginii.

Apoi `merge` elimină dublurile: ia detecțiile în ordinea scorului și o suprimă pe fiecare care se suprapune prea mult cu una mai bună de aceeași clasă. Suprapunerea se măsoară cu **IOS** (intersecția împărțită la aria casetei mai mici), nu cu IoU:

```python
ios = inter / torch.minimum(area[:, None], area[None, :])
suppress = (ios > thr) & (det[:, None, 5] == det[None, :, 5])
```

Motivul: aceeași mașină poate apărea întreagă într-o fereastră și pe jumătate în alta. Caseta mică e cuprinsă complet în cea mare, deci IOS = 1, dar IoU = 0.5 și NMS-ul obișnuit le-ar păstra pe amândouă.

## 6. Ferestrele adaptive

În loc de grilă, ferestrele se pun doar unde detecția preliminară a găsit obiecte ([adaptive.py](../src/pi_p/adaptive.py)):

```python
def window_stages(boxes, width, height, params):
    scale = max(max(width, height) / params.mask_size, 1.0)
    mask = boxes_to_mask(boxes / scale, ceil(width / scale), ceil(height / scale))
    dilated = dilate_mask(mask, params.dilate_kernel)
    rects = mask_to_rects(dilated)
    windows = rects_to_windows(rects, scale, params.margin, params.min_size, width, height)
```

1. **Masca:** casetele preliminare sunt desenate cu alb pe o mască neagră micșorată la 640 px pe latura mare. Pentru 1360×765, `scale = 2.125`. Masca mică face dilatarea și etichetarea mult mai rapide.
2. **Dilatarea** cu un nucleu de 15×15 unește obiectele apropiate într-o singură pată. Nucleul trebuie să fie impar, ca pata să crească la fel în toate direcțiile.
3. **Componentele conexe** (`cv2.connectedComponentsWithStats`): fiecare pată devine un dreptunghi.
4. **Ferestrele:** dreptunghiurile se readuc la scara imaginii, se lărgesc cu o margine de 32 px, se măresc până la cel puțin 640 px și se mută în interiorul imaginii.

Detecția completă ([adaptive_trace](../src/pi_p/adaptive.py)):

- prima trecere, pe imaginea întreagă, la pragul cel mai mic dintre `prelim_conf` și `conf`;
- doar casetele cu scor de cel puțin `prelim_conf` (0.25) devin semințe pentru ferestre, ca zgomotul să nu producă ferestre;
- detecțiile primei treceri peste `conf` se păstrează în rezultat, la fel ca la grilă, unde imaginea întreagă e mereu inclusă;
- detecțiile din ferestre și cele din prima trecere se unesc cu același `merge` ca la grilă.

`pi-eval --mode adaptive` raportează și câte ferestre au fost pe imagine și câți pixeli s-au procesat față de imaginea întreagă, ca să se vadă cât se câștigă față de grilă.

## 7. Evaluarea

Toate modurile se evaluează cu același cod ([evaluate.py](../src/pi_p/evaluate.py) și [metrics.py](../src/pi_p/metrics.py)), pe imaginea întreagă, cu etichetele originale:

- **același prag**, `conf = 0.001`: mAP-ul are nevoie de toată curba precizie-recall, deci și de detecțiile cu scor mic;
- **același plafon de detecții**, 300 pe imagine, sau numărul maxim de obiecte dintr-o imagine a setului, dacă e mai mare, ca variantele cu multe ferestre să nu fie avantajate de numărul mare de casete;
- **aceeași potrivire**: o detecție e corectă dacă are clasa bună și IoU peste prag cu un obiect încă nepotrivit; detecțiile se iau în ordinea scorului, deci un dublu cu scor mai mic devine fals pozitiv;
- **mAP50-95** e media AP-ului pe pragurile de IoU 0.50, 0.55, …, 0.95, iar **mAP50** e doar pragul 0.50. Calculul final al AP-ului e cel din Ultralytics (`ap_per_class`).

Timpul pe imagine se măsoară după 3 imagini de încălzire, iar pe GPU se așteaptă terminarea calculului (`torch.cuda.synchronize`), altfel timpul ar ieși prea mic.

## 8. Întrebări probabile

**De ce nu folosim direct `model.val()` din Ultralytics?** Pentru că lucrează doar pe imaginea întreagă. Variantele cu ferestre produc detecțiile altfel, iar ca să fie comparabile, toate trebuie evaluate cu același cod. `--mode val` rămâne pentru verificări rapide.

**De ce IOS la fuziune și IoU la evaluare?** La fuziune căutăm bucăți ale aceluiași obiect, iar o bucată e conținută în obiectul întreg. La evaluare măsurăm cât de bine se potrivește o casetă cu adevărul, deci contează și cât de mult o depășește.

**Ce se pierde la ferestrele adaptive?** Obiectele pe care prima trecere nu le vede deloc nu primesc fereastră. De aceea e planificată și varianta în care ferestrele adaptive se adaugă grilei.

**De ce CLAHE pe canalul L?** Pe canalele R, G, B separat, CLAHE ar schimba culorile. Pe L schimbă doar luminozitatea.

**De ce imaginea întreagă e inclusă și la grilă?** Obiectele mari pot fi mai mari decât o fereastră, deci în ferestre apar mereu tăiate și sunt aruncate de `drop_cut`. Pe imaginea întreagă micșorată se văd complet.
