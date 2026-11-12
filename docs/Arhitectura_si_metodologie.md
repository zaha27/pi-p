# Detectia pietonilor si a vehiculelor in imagini aeriene: arhitectura sistemului si metodologia experimentala

## 1. Contextul si problema

Proiectul urmareste detectia pietonilor si a vehiculelor in fotografii realizate din drona, pe setul de date VisDrone2019-DET. Detectorul folosit ca punct de plecare este YOLO26n din biblioteca Ultralytics, adaptat prin fine-tuning la cele zece categorii ale setului de date.

Imaginile aeriene difera de fotografiile obisnuite pe care sunt preantrenate detectoarele moderne in trei privinte. Obiectele sunt foarte mici in raport cu imaginea: un pieton poate ocupa doar cativa pixeli pe latime. Obiectele apar in grupuri dense, de exemplu masini parcate sau multimi pe trotuar, si se suprapun frecvent. In plus, conditiile de iluminare variaza mult, de la zi senina la noapte, cu zone subexpuse si contrast scazut.

Un detector precum YOLO redimensioneaza imaginea de intrare la o rezolutie fixa (in configuratia de baza, 640 de pixeli pe latura mare). Pentru o imagine aeriana de rezolutie ridicata, aceasta micsorare face ca obiectele mici sa ajunga la dimensiuni de unul sau doi pixeli, sub pragul la care reteaua le mai poate distinge. Proiectul porneste de la ipoteza ca doua tipuri de prelucrare a imaginii, aplicate inainte si in jurul detectorului, pot reduce aceasta pierdere. Prima este impartirea imaginii in ferestre analizate la rezolutie completa. A doua este imbunatatirea contrastului in imaginile slab iluminate.

## 2. Setul de date

VisDrone2019-DET contine 6471 de imagini de antrenare (train), 548 de imagini de validare (val) si 1610 imagini de test (test-dev), toate cu adnotari realizate manual. Fiecare adnotare descrie un obiect prin chenarul sau (pozitia coltului stanga-sus, latimea si inaltimea), categoria, un scor care arata daca obiectul este luat in considerare, gradul de trunchiere (cat din obiect iese din cadru) si gradul de ocluzie (cat din obiect este acoperit de alte obiecte). Setul de date marcheaza si regiuni ignorate, zone in care obiectele nu au fost adnotate individual; acestea sunt tratate la pregatirea datelor, ca sa nu penalizeze modelul nici la antrenare, nici la evaluare.

Cele trei subseturi au roluri distincte in acest proiect. Subsetul train este folosit exclusiv pentru antrenarea modelului. Subsetul val este folosit pentru toate comparatiile din timpul dezvoltarii si pentru toate deciziile: ce varianta se pastreaza si ce parametri se folosesc. Subsetul test-dev este rulat o singura data, la final, dupa ce toate deciziile au fost fixate. Separarea garanteaza ca rezultatele raportate pe test-dev nu sunt influentate de alegeri facute pe baza lor.

## 3. Arhitectura sistemului

Arhitectura este prezentata in figura de mai jos. Ea descrie drumul unei singure imagini prin sistem, de la intrare pana la detectiile finale si la evaluarea lor. Fiecare componenta poate fi oprita sau inlocuita din fisierul de configurare, iar combinatiile alese formeaza experimentele descrise in sectiunea 5.



### 3.1 Analiza statistica a imaginii

Primul modul caracterizeaza fiecare imagine prin cativa indicatori calculati pe canalul de luminanta: luminozitatea medie, abaterea standard a intensitatilor (o masura a contrastului global) si o estimare a nivelului de zgomot. Pe baza acestor indicatori, imaginea este incadrata intr-una din trei grupe: intuneric, contrast slab sau normal.

Pragurile dintre grupe nu sunt alese arbitrar. Ele sunt stabilite din distributia indicatorilor pe subsetul de antrenare, astfel incat grupele sa corespunda unor situatii reale din date. Rezultatul este un tabel cu cate o linie pentru fiecare imagine, care contine indicatorii si grupa. Grupa are doua intrebuintari: decide daca imaginea primeste imbunatatirea de contrast in varianta adaptiva (sectiunea 3.2) si permite evaluarea separata pe grupe (sectiunea 4). Incadrarea este o eticheta de analiza si nu modifica impartirea setului de date in train, val si test-dev.

### 3.2 Imbunatatirea contrastului

Imbunatatirea contrastului foloseste CLAHE (Contrast Limited Adaptive Histogram Equalization; Zuiderveld, 1994). Spre deosebire de egalizarea globala a histogramei, CLAHE imparte imaginea intr-o grila de regiuni si egalizeaza histograma fiecarei regiuni separat. Valorile din histograma care depasesc o limita (clip limit) sunt taiate si redistribuite uniform, ca sa nu fie amplificat zgomotul in zonele omogene. La final, rezultatele regiunilor vecine sunt imbinate prin interpolare biliniara, ca sa nu apara granite vizibile intre ele.

CLAHE se aplica doar pe canalul de luminanta (de exemplu canalul L din spatiul Lab), iar culorile raman neschimbate, pentru ca detectorul foloseste si informatia de culoare. Sunt studiate doua variante. Varianta uniforma aplica CLAHE tuturor imaginilor. Varianta adaptiva il aplica doar imaginilor din grupele intuneric si contrast slab, iar imaginile normale trec neatinse. Parametrii CLAHE nu sunt optimizati, pentru ca fiecare incercare ar cere o reantrenare a modelului. Se folosesc valori uzuale (o limita de taiere de 2 si o grila de 8×8 regiuni), mentionate explicit in raport.

Imbunatatirea se aplica intotdeauna pe imaginea intreaga, inainte de orice impartire in ferestre, si identic la antrenare si la testare. Aplicarea la nivel de imagine are trei motive. CLAHE este deja o operatie locala, care se adapteaza zonelor diferite din interiorul imaginii. Indicatorii statistici calculati pe o fereastra mica sunt instabili. Iar grupele trebuie definite la nivel de imagine pentru ca evaluarea pe grupe sa fie posibila.

### 3.3 Felierea pe grila fixa

Felierea imparte imaginea in ferestre patrate de aceeasi dimensiune cu intrarea detectorului, asezate pe randuri si coloane. Ferestrele vecine se suprapun partial, astfel incat un obiect taiat de marginea unei ferestre apare intreg in fereastra vecina. Pozitiile ferestrelor sunt alese astfel incat sa acopere toata imaginea. Pe langa ferestre, detectorul primeste si imaginea intreaga, micsorata, care asigura detectia obiectelor mari, ce nu incap intr-o singura fereastra. In implementarea curenta, dimensiunea ferestrei si gradul de suprapunere sunt parametri de configurare.

Ca alternativa la aceasta implementare este evaluata biblioteca SAHI (Slicing Aided Hyper Inference; Akyon et al., 2022), care realizeaza aceeasi idee de feliere la inferenta. Cele doua variante nu se folosesc impreuna in aceeasi rulare: sunt doua realizari ale aceluiasi pas, comparate intre ele.

### 3.4 Detectia preliminara si ferestrele adaptive

Grila fixa analizeaza fiecare zona a imaginii, inclusiv zonele fara obiecte (campuri, acoperisuri, cer), ceea ce costa timp. Aceasta componenta inlocuieste grila cu ferestre plasate doar acolo unde exista obiecte, in doi pasi.

Primul pas, detectia preliminara, ruleaza detectorul pe imaginea micsorata si obtine o estimare aproximativa a pozitiilor obiectelor. Ideea de a cauta intai grosier si de a analiza detaliat doar zonele promitatoare este preluata din QueryDet (Yang et al., 2022). Acolo, cautarea grosiera se face insa in interiorul retelei neuronale, pe hartile de caracteristici de rezolutie mica, si este antrenata impreuna cu detectorul.

Al doilea pas transforma detectiile preliminare in ferestre prin operatii clasice de prelucrare a imaginilor. Detectiile sunt desenate ca regiuni albe pe o imagine binara neagra (masca). Masca este dilatata morfologic, astfel incat regiunile apropiate sa se uneasca. Fiecare componenta conexa ramasa reprezinta un grup de obiecte, iar dreptunghiul care o incadreaza devine o fereastra de analiza. Ideea de a grupa obiectele in regiuni si de a analiza separat fiecare regiune vine din ClusDet (Yang et al., 2019), unde regiunile sunt propuse de o retea separata, antrenata in acest scop. O abordare apropiata, bazata tot pe detectii initiale, este Cascaded Zoom-in (Meethal et al., 2023). Contributia proiectului nu este ideea in sine, ci realizarea ei fara nicio retea suplimentara, numai cu operatii clasice: binarizare, dilatare si etichetarea componentelor conexe.

Metoda are doua limite cunoscute. Obiectele pe care detectia preliminara nu le vede deloc pe imaginea micsorata nu primesc fereastra. Iar gradul de dilatare trebuie reglat: o dilatare prea mica produce multe ferestre mici, iar una prea mare produce grupuri foarte extinse, care trebuie la randul lor impartite. Pentru prima limita este prevazuta varianta in care ferestrele adaptive se adauga grilei, care ramane ca plasa de siguranta.

### 3.5 Detectia pe ferestre

Detectorul antrenat ruleaza pe fiecare fereastra, indiferent de modul in care au fost alese ferestrele. Chenarele obtinute sunt exprimate in coordonatele ferestrei, asa ca sunt mutate in coordonatele imaginii intregi prin adunarea pozitiei coltului stanga-sus al ferestrei. De exemplu, un chenar gasit intre punctele (100, 50) si (140, 90) intr-o fereastra care incepe la (512, 0) devine, in imaginea intreaga, chenarul dintre (612, 50) si (652, 90).

### 3.6 Fuziunea detectiilor

Din cauza suprapunerii dintre ferestre, acelasi obiect poate fi detectat de mai multe ori, iar un obiect taiat de marginea unei ferestre poate fi detectat ca obiect partial. Fuziunea rezolva ambele situatii in doi pasi.

In primul pas sunt eliminate detectiile taiate. O detectie este eliminata daca atinge, la o distanta de cel mult doi pixeli, o margine a ferestrei aflata in interiorul imaginii. Marginile care coincid cu marginea imaginii nu sunt luate in considerare, pentru ca acolo obiectul este taiat de cadrul fotografiei, nu de fereastra. Obiectul eliminat astfel apare intreg in fereastra vecina, datorita suprapunerii.

In al doilea pas sunt eliminate dublurile, printr-o varianta de suprimare a non-maximelor (NMS). Detectiile sunt ordonate descrescator dupa scorul de incredere. O detectie este eliminata daca se suprapune puternic cu o detectie de aceeasi clasa si cu scor mai mare. Masura de suprapunere folosita nu este IoU (Intersection over Union), ci IOS (Intersection over Smaller):

IoU(A, B) = aria(A ∩ B) / aria(A ∪ B)

IOS(A, B) = aria(A ∩ B) / min(aria(A), aria(B))

Diferenta conteaza cand un chenar mic este cuprins aproape in intregime intr-unul mare, de exemplu o detectie partiala a unei masini in interiorul detectiei intregi. IoU are in acest caz o valoare mica, iar dublura ar fi pastrata. IOS are o valoare apropiata de 1, iar dublura este eliminata. Pragul implicit de suprimare este 0,5.

### 3.7 Antrenarea modelului

Modelul porneste de la greutatile YOLO26n preantrenate si este adaptat prin fine-tuning pe subsetul de antrenare. Sunt pregatite doua configuratii de antrenare: una pe imagini intregi si una pe ferestre suprapuse taiate din imaginile de antrenare. Atunci cand se foloseste imbunatatirea contrastului, ea este aplicata imaginilor de antrenare inainte de taierea ferestrelor, exact ca la testare, ca modelul sa fie antrenat pe acelasi tip de imagine pe care il va primi la utilizare.

Antrenarea nu face parte din drumul unei imagini prin sistem. Ea are loc o singura data, inainte de orice testare, iar rezultatul ei, fisierul de greutati, este folosit de componentele de detectie.

## 4. Evaluarea

Detectiile finale sunt comparate cu adnotarile manuale ale imaginii intregi, dupa reasamblarea si fuziunea detectiilor din ferestre. Evaluarea pe ferestre separate ar produce valori care nu pot fi comparate cu cele obtinute pe imagini intregi.

O detectie este considerata corecta daca are aceeasi clasa ca un obiect adnotat si IoU-ul cu chenarul acestuia depaseste un prag. Pentru fiecare clasa se calculeaza precizia medie (AP), adica aria de sub curba precizie–recall obtinuta prin varierea pragului de incredere. Media AP pe clase este mAP. Sunt raportate doua variante uzuale: mAP50, cu pragul IoU de 0,5, si mAP50-95, media valorilor mAP pentru praguri IoU de la 0,5 la 0,95 cu pasul 0,05. Aceasta din urma recompenseaza si precizia localizarii.

Pentru ca rularile sa fie comparabile, toate sunt evaluate in aceleasi conditii. Se folosesc aceleasi imagini, acelasi prag de incredere si aceleasi praguri de fuziune, iar adnotarile de referinta sunt cele ale imaginii intregi. Pragul de incredere trebuie sa fie acelasi si scazut pentru toate variantele, pentru ca mAP depinde de detectiile cu scor mic. Un prag mai mare la una dintre variante ar dezavantaja-o artificial.

Pe langa scorul pe intregul set, mAP este calculat separat pe fiecare grupa de imagini stabilita in sectiunea 3.1. Astfel se poate verifica daca o metoda ajuta in special imaginile intunecate sau slab contrastate, chiar daca efectul pe intregul set este mic. Este masurat si timpul mediu de procesare pe imagine, pe acelasi echipament, pentru ca felierea creste numarul de rulari ale detectorului. Ca extensie, poate fi raportata si precizia pe categorii de marime a obiectelor, dupa conventia COCO, care ar sustine direct afirmatiile despre obiectele mici.

## 5. Metodologia experimentala

Experimentele urmeaza schema din figura de mai jos. Ele sunt organizate in etape: la fiecare etapa se adauga o singura componenta, iar ea este pastrata doar daca imbunatateste rezultatul pe setul de validare. Astfel, fiecare comparatie difera printr-un singur element, iar diferenta de rezultat poate fi atribuita acelui element.



In etapa 1 se stabileste modul de impartire a imaginii. Sunt comparate trei variante. Baseline-ul foloseste modelul antrenat pe imagini intregi, testat pe imaginea intreaga, si este referinta tuturor comparatiilor. Varianta SAHI foloseste acelasi model, testat pe ferestre prin biblioteca SAHI, fara reantrenare. Varianta cu feliere pe grila fixa foloseste modelul antrenat pe ferestre, testat pe ferestre, cu fuziunea descrisa in sectiunea 3.6. Comparatia dintre baseline si SAHI arata efectul felierii la testare. Comparatia dintre SAHI si felierea pe grila arata efectul suplimentar al antrenarii pe ferestre. Varianta cu cel mai bun mAP pe setul de validare devine configuratia selectata.

In etapa 2 se evalueaza imbunatatirea contrastului, pe configuratia selectata in etapa 1. Pentru fiecare dintre cele doua variante, uniforma si adaptiva, CLAHE este aplicat imaginilor de antrenare inainte de taiere, modelul este reantrenat, iar testarea pe setul de validare se face in acelasi mod ca in configuratia selectata. CLAHE este pastrat doar daca imbunatateste rezultatul. Un rezultat negativ, care ar arata ca imbunatatirea contrastului nu ajuta pe acest set de date, este si el o concluzie valabila si este raportat ca atare.

Etapa 3 evalueaza ferestrele adaptive pe configuratia selectata in etapa 2, in doua variante: in locul grilei fixe si impreuna cu ea (a doua varianta este planificata).

Dupa ce toate deciziile au fost luate pe setul de validare, toate variantele sunt rulate o singura data pe test-dev. Pentru ca aceasta rulare nu cere reantrenare, costul ei este doar timpul de inferenta. Tabelul final contine, pentru fiecare varianta, valorile pe setul de validare, care justifica alegerile, si valorile pe test-dev, care sunt rezultatele raportate.

| Etapa | Varianta | Antrenare | Testare |
|---|---|---|---|
| 1 | Baseline | imagini intregi | imagine intreaga |
| 1 | SAHI | imagini intregi | ferestre (biblioteca SAHI) |
| 1 | Feliere pe grila fixa | ferestre | ferestre si imaginea intreaga micsorata |
| 2 | CLAHE uniform | ca in configuratia selectata, cu CLAHE pe toate imaginile | la fel |
| 2 | CLAHE adaptiv | ca in configuratia selectata, cu CLAHE pe grupele care il cer | la fel |
| 3 | Ferestre adaptive | ca in configuratia selectata | ferestre adaptive |
| 3 (planificat) | Ferestre adaptive si grila | ca in configuratia selectata | ferestre adaptive si grila |

Metodologia are o limita asumata. Selectia pe etape nu verifica toate combinatiile posibile. De exemplu, nu arata daca CLAHE ar fi avut un efect mai mare pe o varianta de impartire care nu a fost selectata in etapa 1. Testarea tuturor combinatiilor ar multiplica numarul de antrenari si ramane o directie pentru continuarea proiectului.

## 6. Starea implementarii

Sunt implementate pregatirea datelor, analiza statistica a imaginilor, imbunatatirea contrastului in cele doua variante, configuratiile de antrenare pe imagini intregi si pe ferestre, taierea ferestrelor pentru antrenare, inferenta pe ferestre cu fuziunea detectiilor (eliminarea detectiilor taiate si suprimarea dublurilor prin IOS), integrarea bibliotecii SAHI, detectia preliminara cu ferestrele adaptive si un evaluator comun, care calculeaza mAP-ul pe imaginea intreaga in aceleasi conditii pentru toate variantele. Urmeaza rularea experimentelor din sectiunea 5 si varianta in care ferestrele adaptive se adauga grilei. Legatura dintre comenzi si module este descrisa in [pipeline.md](pipeline.md), iar fiecare etapa este explicata pas cu pas in [pipeline/README.md](pipeline/README.md).

Din experimente sunt rulate pana acum: etapa 1 (imagini intregi fata de felii) cu antrenari complete de 100 de epoci, o prima reglare a ferestrelor adaptive si reantrenarile cu CLAHE uniform si adaptiv, evaluate pe val; testul final pe test-dev nu este inca rulat. Valorile sunt in [rezultate.md](rezultate.md).

## Bibliografie

Akyon, F. C., Altinuc, S. O., Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. IEEE International Conference on Image Processing (ICIP).

Meethal, A., Granger, E., Pedersoli, M. (2023). Cascaded Zoom-in Detector for High Resolution Aerial Images. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). arXiv:2303.08747.

Yang, C., Huang, Z., Wang, N. (2022). QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yang, F., Fan, H., Chu, P., Blasch, E., Ling, H. (2019). Clustered Object Detection in Aerial Images. IEEE/CVF International Conference on Computer Vision (ICCV).

Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., Ling, H. (2022). Detection and Tracking Meet Drones Challenge. IEEE Transactions on Pattern Analysis and Machine Intelligence. arXiv:2001.06303.

Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In P. Heckbert (ed.), Graphics Gems IV, Academic Press, 474–485.

Ultralytics (2026). Ultralytics YOLO, documentatie si cod sursa. https://docs.ultralytics.com
