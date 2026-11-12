# Codul, explicat

Ghid pentru prezentare: ideile importante din cod, in ordinea pipeline-ului, fiecare cu fragmentul care o implementeaza si un exemplu. Legatura dintre comenzi e in [pipeline.md](pipeline.md).

## 1. Datele

Etichetele sunt in formatul YOLO: un fisier `.txt` pentru fiecare imagine, cu cate un rand `clasa x_centru y_centru latime inaltime` pentru fiecare obiect. Coordonatele sunt impartite la dimensiunile imaginii, deci sunt intre 0 si 1. Functiile de citire si scriere sunt in [data.py](../src/pi_p/data.py), iar restul modulelor le folosesc doar pe acestea.

VisDrone marcheaza unele zone ca „ignored” (multimi pe care adnotatorii nu le-au etichetat) si are o clasa „others” pe care nu o folosim. Daca raman in imagine, modelul e penalizat pentru obiecte reale pe care le gaseste acolo. `pi-prepare` le acopera cu gri, dar lasa neatinse obiectele etichetate care se suprapun cu ele ([prepare.py](../src/pi_p/prepare.py)):

```python
def mask_ignored(img, keep, ignore):
    m = np.zeros(img.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in ignore:
        m[y1:y2, x1:x2] = True
    for x1, y1, x2, y2 in keep:
        m[y1:y2, x1:x2] = False
    img[m] = FILL  # 114, griul folosit si de YOLO la completarea imaginilor
```

## 2. Analiza statistica

Pentru fiecare imagine se calculeaza media si abaterea standard a canalului L din spatiul de culoare LAB, adica luminozitatea si contrastul ([stats.py](../src/pi_p/stats.py)). Imaginea e citita micsorata de 4 ori, pentru ca media si abaterea abia se schimba, iar citirea e mult mai rapida.

```python
def group(brightness, contrast, dark=70, low_contrast=35):
    if brightness < dark:
        return "dark"
    if contrast < low_contrast:
        return "low_contrast"
    return "normal"
```

Pragurile 70 si 35 se pot schimba cu `--dark` si `--low-contrast`, iar histogramele din `stats.png` arata unde cad fata de distributie. Rezultatul pe VisDrone:

| Split | normal | dark | low_contrast |
|---|---|---|---|
| train | 4736 | 1139 | 596 |
| val | 431 | 13 | 104 |
| test-dev | 1032 | 501 | 77 |

Pe val sunt doar 13 imagini intunecate, asa ca mAP-ul pe grupa `dark` din val e foarte zgomotos. Concluziile despre imaginile intunecate trebuie verificate pe test-dev, unde sunt 501.

## 3. CLAHE

CLAHE egalizeaza histograma pe bucati de imagine (grila de 8×8) si limiteaza cat poate creste contrastul (`clip=2.0`), ca sa nu amplifice zgomotul. Se aplica doar pe canalul L, deci culorile raman aceleasi ([enhance.py](../src/pi_p/enhance.py)):

```python
def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def needs_clahe(name, mode, groups):
    return mode == "uniform" or groups.get(name) in {"dark", "low_contrast"}
```

Varianta `uniform` aplica CLAHE pe toate imaginile, iar `adaptive` doar pe cele din grupele slabe. Etichetele se copiaza neschimbate, pentru ca CLAHE nu muta pixelii.

## 4. Grila de ferestre

Ferestrele au 640 px si se suprapun 20%, deci pasul e 512 px. Ultima fereastra de pe fiecare axa e lipita de marginea imaginii, ca sa nu ramana o banda neacoperita ([slicing.py](../src/pi_p/slicing.py)):

```python
def starts(length, size, step):
    if length <= size:
        return [0]
    return [*range(0, length - size, step), length - size]
```

Exemplu: o imagine VisDrone de 1360×765 primeste ferestrele care incep la x ∈ {0, 512, 720} si y ∈ {0, 125}, adica 6 ferestre.

La antrenarea pe felii, `pi-tile` decupeaza aceleasi ferestre si pastreaza un obiect doar daca cel putin jumatate din el se vede in fereastra (`--min-visibility 0.5`). Altfel, modelul ar invata sa recunoasca bucati mici de masini ca masini intregi. Ferestrele fara obiecte se pastreaza doar in proportie de 10% (`--empty 0.1`), ca modelul sa vada si fundal, dar fara sa fie coplesit de el.

## 5. Fuziunea detectiilor

Detectiile dintr-o fereastra sunt in coordonatele ferestrei. Inainte de unire se fac doi pasi ([predict.py](../src/pi_p/predict.py)):

```python
def to_image(det, win, w, h):
    return drop_cut(det, win, w, h) + torch.tensor([win[0], win[1], win[0], win[1], 0, 0])
```

1. `drop_cut` arunca detectiile care ating o margine a ferestrei aflata in interiorul imaginii: obiectul e taiat acolo si apare intreg in fereastra vecina. O detectie lipita de marginea reala a imaginii ramane. De exemplu, in fereastra care incepe la x = 512, o caseta cu x1 = 0 e aruncata.
2. Se aduna coltul stanga-sus al ferestrei, ca sa ajunga in coordonatele imaginii.

Apoi `merge` elimina dublurile: ia detectiile in ordinea scorului si o suprima pe fiecare care se suprapune prea mult cu una mai buna de aceeasi clasa. Suprapunerea se masoara cu **IOS** (intersectia impartita la aria casetei mai mici), nu cu IoU:

```python
ios = inter / torch.minimum(area[:, None], area[None, :])
suppress = (ios > thr) & (det[:, None, 5] == det[None, :, 5])
```

Motivul: aceeasi masina poate aparea intreaga intr-o fereastra si pe jumatate in alta. Caseta mica e cuprinsa complet in cea mare, deci IOS = 1, dar IoU = 0.5 si NMS-ul obisnuit le-ar pastra pe amandoua.

## 6. Ferestrele adaptive

In loc de grila, ferestrele se pun doar unde detectia preliminara a gasit obiecte ([adaptive.py](../src/pi_p/adaptive.py)):

```python
def window_stages(boxes, width, height, params):
    scale = max(max(width, height) / params.mask_size, 1.0)
    mask = boxes_to_mask(boxes / scale, ceil(width / scale), ceil(height / scale))
    dilated = dilate_mask(mask, params.dilate_kernel)
    rects = mask_to_rects(dilated)
    windows = rects_to_windows(rects, scale, params.margin, params.min_size, width, height)
```

1. **Masca:** casetele preliminare sunt desenate cu alb pe o masca neagra micsorata la 640 px pe latura mare. Pentru 1360×765, `scale = 2.125`. Masca mica face dilatarea si etichetarea mult mai rapide.
2. **Dilatarea** cu un nucleu de 15×15 uneste obiectele apropiate intr-o singura pata. Nucleul trebuie sa fie impar, ca pata sa creasca la fel in toate directiile.
3. **Componentele conexe** (`cv2.connectedComponentsWithStats`): fiecare pata devine un dreptunghi.
4. **Ferestrele:** dreptunghiurile se readuc la scara imaginii, se largesc cu o margine de 32 px, se maresc pana la cel putin 640 px si se muta in interiorul imaginii.

Detectia completa ([adaptive_trace](../src/pi_p/adaptive.py)):

- prima trecere, pe imaginea intreaga, la pragul cel mai mic dintre `prelim_conf` si `conf`;
- doar casetele cu scor de cel putin `prelim_conf` (0.25) devin seminte pentru ferestre, ca zgomotul sa nu produca ferestre;
- detectiile primei treceri peste `conf` se pastreaza in rezultat, la fel ca la grila, unde imaginea intreaga e mereu inclusa;
- detectiile din ferestre si cele din prima trecere se unesc cu acelasi `merge` ca la grila.

`pi-eval --mode adaptive` raporteaza si cate ferestre au fost pe imagine si cati pixeli s-au procesat fata de imaginea intreaga, ca sa se vada cat se castiga fata de grila.

## 7. Evaluarea

Toate modurile se evalueaza cu acelasi cod ([evaluate.py](../src/pi_p/evaluate.py) si [metrics.py](../src/pi_p/metrics.py)), pe imaginea intreaga, cu etichetele originale:

- **acelasi prag**, `conf = 0.001`: mAP-ul are nevoie de toata curba precizie-recall, deci si de detectiile cu scor mic;
- **acelasi plafon de detectii**, 300 pe imagine, sau numarul maxim de obiecte dintr-o imagine a setului, daca e mai mare, ca variantele cu multe ferestre sa nu fie avantajate de numarul mare de casete;
- **aceeasi potrivire**: o detectie e corecta daca are clasa buna si IoU peste prag cu un obiect inca nepotrivit; detectiile se iau in ordinea scorului, deci un dublu cu scor mai mic devine fals pozitiv;
- **mAP50-95** e media AP-ului pe pragurile de IoU 0.50, 0.55, …, 0.95, iar **mAP50** e doar pragul 0.50. Calculul final al AP-ului e cel din Ultralytics (`ap_per_class`).

Timpul pe imagine se masoara dupa 3 imagini de incalzire, iar pe GPU se asteapta terminarea calculului (`torch.cuda.synchronize`), altfel timpul ar iesi prea mic.

## 8. Intrebari probabile

**De ce nu folosim direct `model.val()` din Ultralytics?** Pentru ca lucreaza doar pe imaginea intreaga. Variantele cu ferestre produc detectiile altfel, iar ca sa fie comparabile, toate trebuie evaluate cu acelasi cod. `--mode val` ramane pentru verificari rapide.

**De ce IOS la fuziune si IoU la evaluare?** La fuziune cautam bucati ale aceluiasi obiect, iar o bucata e continuta in obiectul intreg. La evaluare masuram cat de bine se potriveste o caseta cu adevarul, deci conteaza si cat de mult o depaseste.

**Ce se pierde la ferestrele adaptive?** Obiectele pe care prima trecere nu le vede deloc nu primesc fereastra. De aceea e planificata si varianta in care ferestrele adaptive se adauga grilei.

**De ce CLAHE pe canalul L?** Pe canalele R, G, B separat, CLAHE ar schimba culorile. Pe L schimba doar luminozitatea.

**De ce imaginea intreaga e inclusa si la grila?** Obiectele mari pot fi mai mari decat o fereastra, deci in ferestre apar mereu taiate si sunt aruncate de `drop_cut`. Pe imaginea intreaga micsorata se vad complet.
