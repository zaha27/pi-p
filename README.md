# pi-personal-project

Detectia pietonilor si a vehiculelor in imagini filmate din drona (VisDrone), cu YOLO26n. Proiectul studiaza doua prelucrari de imagine puse in jurul detectorului: taierea imaginii in ferestre analizate la rezolutie completa si imbunatatirea contrastului cu CLAHE.

## Unde gasesti ce

| Vrei sa... | Citeste |
|---|---|
| intelegi fiecare etapa, cu scheme si exemple | [docs/pipeline/README.md](docs/pipeline/README.md) |
| vezi pe o pagina cum se leaga comenzile | [docs/pipeline.md](docs/pipeline.md) |
| citesti codul explicat functie cu functie | [docs/cod.md](docs/cod.md) |
| vezi ce rezultate au iesit | [docs/rezultate.md](docs/rezultate.md) |
| folosesti modelele antrenate | [weights/README.md](weights/README.md) |
| vezi arhitectura intr-o singura schema | [docs/arhitectura.md](docs/arhitectura.md) |
| citesti descrierea completa si metodologia | [docs/Arhitectura_si_metodologie.md](docs/Arhitectura_si_metodologie.md) |
| vezi planul experimentelor | [docs/benchmarkuri.md](docs/benchmarkuri.md) |
| vezi diagramele UML, inclusiv aplicatia web | [docs/uml.md](docs/uml.md) |

## Pornire rapida

```bash
uv sync                                  # instaleaza dependentele
uv run pytest                            # ruleaza testele
uv run pi-prepare                        # descarca si pregateste VisDrone
uv run pi-train configs/smoke.yaml       # antrenare de proba, ~1 minut
```

Toate comenzile, in ordinea pipeline-ului, sunt in [docs/pipeline/README.md](docs/pipeline/README.md#toate-comenzile-in-ordine).

## Structura

```
src/pi_p/           pipeline-ul: date, CLAHE, feliere, antrenare, detectie, evaluare
configs/            parametrii antrenarilor si ai ferestrelor adaptive
weights/           modelele antrenate (.pt)
results/           rezultate si figuri salvate
tests/              teste pentru fiecare etapa
docs/               documentatia
infrastructure/     aplicatia web: server FastAPI, dashboard, Docker
```

## Licenta

Toate drepturile rezervate. Codul, documentatia, figurile si rezultatele nu pot fi copiate, modificate, redistribuite sau folosite in alte proiecte fara acordul scris al autorilor. Detalii in [LICENSE](LICENSE).

## Analiza literaturii de specialitate (state-of-the-art)

| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicatie / Domeniu<sup>1</sup> | Tehnologii utilizate<sup>2</sup> | Metodologie / Abordare<sup>3</sup> | Rezultate<sup>4</sup> | Limitari<sup>5</sup> | Comentarii suplimentare<sup>6</sup> |
|---|---|---|---|---|---|---|---|---|
| 1 | F. C. Akyon, S. O. Altinuc, A. Temizel / 2022 (ICIP) | Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI) | Detectia obiectelor mici in imagini de inalta rezolutie (aeriene, drone, satelit) | Python, PyTorch; detectoare FCOS, VFNet, TOOD (MMDetection); biblioteca open-source `sahi` compatibila cu YOLO, Detectron2, HuggingFace | Imaginea e impartita in felii (slices) suprapuse, detectorul ruleaza pe fiecare felie, plus o trecere pe imaginea intreaga; predictiile sunt mapate inapoi si unite prin NMS/NMM. Fine-tuning pe felii (slicing aided fine-tuning) | Pe VisDrone si xView: +6.8, +5.1, +5.3 AP pentru FCOS, VFNet, TOOD doar la inferenta; cu fine-tuning, cumulat +12.7, +13.4, +14.5 AP | Timpul de inferenta creste cu numarul de felii; obiectele taiate la marginea feliilor pot genera cutii duplicate sau fragmentate; dimensiunea feliei si suprapunerea trebuie reglate pe dataset | Felii independente, deci se pot procesa in paralel (batch pe GPU sau mai multi workeri), ceea ce compenseaza costul de inferenta; relevant direct pentru analiza video aeriana din proiect |
| 2 | P. Zhu, L. Wen, X. Bian, H. Ling, Q. Hu / 2018 (arXiv) | Vision Meets Drones: A Challenge (VisDrone2018) | Benchmark pentru detectie si urmarire de obiecte din imagini si video capturate cu drone | Drone cu camere de diverse tipuri; adnotari manuale (bounding box, categorie, ocluzie, trunchiere) | Colectare in 14 orase din China, in conditii variate (scene, densitate, iluminare, altitudine); 4 task-uri: detectie in imagini, detectie in video, urmarire single-object, urmarire multi-object | 10.209 imagini statice, 263 clipuri video, 179.264 cadre in total, peste 2,5 milioane de obiecte adnotate; 10 categorii (pietoni, masini, autobuze, biciclete etc.) | Doar zone urbane din China; obiecte foarte mici si dense, greu de adnotat consistent; lucrarea descrie datasetul, nu propune o metoda | Dataset de referinta pentru detectia de obiecte mici din drone, folosit si in evaluarea SAHI (randul 1); candidat principal pentru antrenarea si testarea modelului din proiect |
| 3 | C. Yang, Z. Huang, N. Wang / 2022 (CVPR) | QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection | Detectia rapida a obiectelor mici in imagini de inalta rezolutie (COCO, imagini de la drone) | Detectoare cu feature pyramid (FPN, de ex. RetinaNet); convolutii sparse; PyTorch, cod open-source | Cascade Sparse Query (CSQ): pe nivelurile de rezolutie mica ale piramidei, un query head estimeaza unde pot fi obiecte mici; pe nivelurile de rezolutie mare, capul de detectie calculeaza doar in acele pozitii, in cascada pana la nivelul cel mai fin | COCO: +1.0 mAP, +2.0 mAP pe obiecte mici, inferenta la rezolutie mare de ~3x mai rapida; VisDrone: state-of-the-art la publicare, ~2.3x mai rapid | Necesita modificarea arhitecturii si reantrenare; convolutiile sparse au suport limitat la export (ONNX/TensorRT); costul pe cadru variaza cu numarul de query-uri, deci timpul de inferenta e mai putin previzibil | Preluam ideea cautarii grosiere urmate de analiza detaliata, dar in afara retelei si fara reantrenare: detectorul ruleaza intai pe imaginea micsorata, iar din detectiile preliminare se construiesc ferestre de analiza prin operatii clasice (masca binara, dilatare morfologica, componente conexe). Decizie: nu folosim convolutii sparse; sparsitatea se muta de la nivel de pixel (convolutii sparse) la nivel de fereastra: ferestrele selectate trec prin convolutii dense obisnuite, eficiente pe GPU si exportabile in ONNX/TensorRT, fara biblioteci speciale (`spconv`). Pretul: obiectele pe care pasul grosier nu le vede nu primesc fereastra, de aceea e prevazuta si varianta in care ferestrele adaptive se adauga grilei fixe. Benchmark pe VisDrone (AP vs. timp pe imagine) intre grila fixa, ferestrele adaptive si ferestrele adaptive impreuna cu grila |
| 4 | F. Yang, H. Fan, P. Chu, E. Blasch, H. Ling / 2019 (ICCV) | Clustered Object Detection in Aerial Images (ClusDet) | Detectia obiectelor mici si distribuite neuniform in imagini aeriene | Retea end-to-end cu trei sub-retele: CPNet, ScaleNet, DetecNet | CPNet propune regiuni unde obiectele formeaza clustere; ScaleNet estimeaza scara obiectelor din fiecare regiune, iar regiunea e redimensionata; DetecNet detecteaza pe regiunile normalizate si pe imaginea intreaga, rezultatele se unesc prin NMS | Evaluat pe VisDrone, UAVDT si DOTA; mai putine regiuni (chips) procesate decat la taierea uniforma, deci timp de rulare mai mic, si precizie mai buna pe obiecte mici | Arhitectura proprie, cu sub-retele care trebuie antrenate; nu se poate pune peste un detector existent; numarul si marimea regiunilor variaza de la cadru la cadru, deci batch-ul pe GPU e greu de planificat | Selectie a zonelor la nivel de imagine, dar invatata. Varianta noastra pastreaza ideea gruparii obiectelor in regiuni analizate separat, dar regiunile se obtin din detectiile preliminare prin dilatare morfologica si etichetarea componentelor conexe, fara retele antrenate in plus. Ferestrele au marime variabila, la fel ca la ClusDet |
| 5 | Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang / 2022 (ECCV) | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | Urmarirea mai multor obiecte (multi-object tracking) in video | Detector YOLOX; filtru Kalman pentru predictia miscarii; asociere prin IoU si algoritmul ungar; PyTorch, cod open-source | Asociere in doua etape: intai detectiile cu scor mare sunt potrivite cu traiectoriile existente, apoi cele cu scor mic sunt potrivite cu traiectoriile ramase, ca sa recupereze obiecte ocluzionate sau neclare; detectiile cu scor mic nepotrivite sunt aruncate ca fundal | MOT17 test: 80.3 MOTA, 77.3 IDF1, 63.1 HOTA, 30 FPS pe un V100; state-of-the-art si pe MOT20, HiEve, BDD100K; aplicata pe 9 trackere, asocierea creste IDF1 cu ~1-10 puncte | Depinde de calitatea detectorului; doar miscare si IoU, fara descriptori de aspect, deci identitatile se pot schimba la ocluzii lungi; testat pe camere fixe sau de la sol, nu pe drone, unde camera se misca | Acopera partea video a proiectului. Pastrarea detectiilor cu scor mic conteaza la drone, unde obiectele mici au des incredere mica. Pozitiile prezise de tracker ghideaza alegerea feliilor la cadrul urmator, iar grila completa ruleaza doar periodic |
| 6 | K. Zuiderveld / 1994 (Graphics Gems IV, pp. 474-485) | Contrast Limited Adaptive Histogram Equalization (CLAHE) | Imbunatatirea contrastului local in imagini (initial imagistica medicala); preprocesare pentru imagini aeriene cu ceata, umbre sau lumina slaba | Algoritm clasic de prelucrare a imaginilor; implementari in OpenCV (`cv2.createCLAHE`, varianta CUDA `cv2.cuda.createCLAHE`) si Albumentations | Imaginea e impartita in tile-uri; pentru fiecare tile, histograma e taiata la un prag (clip limit), surplusul e redistribuit uniform, apoi se aplica egalizarea; tranzitia intre tile-uri se face prin interpolare biliniara. Pe imagini color se aplica doar pe canalul de luminozitate (L din LAB) | Contrast local imbunatatit fara amplificarea zgomotului din zonele uniforme, problema principala a AHE (Pizer et al., 1987); cost mic, de ordinul milisecundelor pe cadru | Parametrii (clip limit, marimea tile-urilor) depind de imagine; pe imagini deja bune poate amplifica zgomotul fara castig; un detector antrenat fara CLAHE poate pierde precizie daca CLAHE apare doar la inferenta | Ultralytics YOLO foloseste deja CLAHE la antrenare, ca augmentare prin Albumentations, dar cu probabilitate mica (`A.CLAHE(p=0.01)`) si doar daca pachetul e instalat. Benchmark pe VisDrone, mai ales pe scenele de noapte si cu ceata: fara CLAHE, CLAHE la inferenta, CLAHE la antrenare si inferenta, CLAHE aplicat doar pe cadrele cu contrast slab (decis dupa histograma). Paralela cu SAHI: ambele impart imaginea spatial si proceseaza bucatile independent |
