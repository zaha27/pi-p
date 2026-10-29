# Detecția pietonilor și a vehiculelor în imagini aeriene: arhitectura sistemului și metodologia experimentală

## 1. Contextul și problema

Proiectul urmărește detecția pietonilor și a vehiculelor în fotografii realizate din dronă, pe setul de date VisDrone2019-DET. Detectorul folosit ca punct de plecare este YOLO26n din biblioteca Ultralytics, adaptat prin fine-tuning la cele zece categorii ale setului de date.

Imaginile aeriene diferă de fotografiile obișnuite pe care sunt preantrenate detectoarele moderne în trei privințe. Obiectele sunt foarte mici în raport cu imaginea: un pieton poate ocupa doar câțiva pixeli pe lățime. Obiectele apar în grupuri dense, de exemplu mașini parcate sau mulțimi pe trotuar, și se suprapun frecvent. În plus, condițiile de iluminare variază mult, de la zi senină la noapte, cu zone subexpuse și contrast scăzut.

Un detector precum YOLO redimensionează imaginea de intrare la o rezoluție fixă (în configurația de bază, 640 de pixeli pe latura mare). Pentru o imagine aeriană de rezoluție ridicată, această micșorare face ca obiectele mici să ajungă la dimensiuni de unul sau doi pixeli, sub pragul la care rețeaua le mai poate distinge. Proiectul pornește de la ipoteza că două tipuri de prelucrare a imaginii, aplicate înainte și în jurul detectorului, pot reduce această pierdere. Prima este împărțirea imaginii în ferestre analizate la rezoluție completă. A doua este îmbunătățirea contrastului în imaginile slab iluminate.

## 2. Setul de date

VisDrone2019-DET conține 6471 de imagini de antrenare (train), 548 de imagini de validare (val) și 1610 imagini de test (test-dev), toate cu adnotări realizate manual. Fiecare adnotare descrie un obiect prin chenarul său (poziția colțului stânga-sus, lățimea și înălțimea), categoria, un scor care arată dacă obiectul este luat în considerare, gradul de trunchiere (cât din obiect iese din cadru) și gradul de ocluzie (cât din obiect este acoperit de alte obiecte). Setul de date marchează și regiuni ignorate, zone în care obiectele nu au fost adnotate individual; acestea sunt tratate la pregătirea datelor, ca să nu penalizeze modelul nici la antrenare, nici la evaluare.

Cele trei subseturi au roluri distincte în acest proiect. Subsetul train este folosit exclusiv pentru antrenarea modelului. Subsetul val este folosit pentru toate comparațiile din timpul dezvoltării și pentru toate deciziile: ce variantă se păstrează și ce parametri se folosesc. Subsetul test-dev este rulat o singură dată, la final, după ce toate deciziile au fost fixate. Separarea garantează că rezultatele raportate pe test-dev nu sunt influențate de alegeri făcute pe baza lor.

## 3. Arhitectura sistemului

Arhitectura este prezentată în figura de mai jos. Ea descrie drumul unei singure imagini prin sistem, de la intrare până la detecțiile finale și la evaluarea lor. Componentele cu contur continuu fac parte din prototip; cele cu contur punctat sunt planificate pentru etapele următoare. Fiecare componentă poate fi oprită sau înlocuită din fișierul de configurare, iar combinațiile alese formează experimentele descrise în secțiunea 5.



### 3.1 Analiza statistică a imaginii

Primul modul caracterizează fiecare imagine prin câțiva indicatori calculați pe canalul de luminanță: luminozitatea medie, abaterea standard a intensităților (o măsură a contrastului global) și o estimare a nivelului de zgomot. Pe baza acestor indicatori, imaginea este încadrată într-una din trei grupe: întuneric, contrast slab sau normal.

Pragurile dintre grupe nu sunt alese arbitrar. Ele sunt stabilite din distribuția indicatorilor pe subsetul de antrenare, astfel încât grupele să corespundă unor situații reale din date. Rezultatul este un tabel cu câte o linie pentru fiecare imagine, care conține indicatorii și grupa. Grupa are două întrebuințări: decide dacă imaginea primește îmbunătățirea de contrast în varianta adaptivă (secțiunea 3.2) și permite evaluarea separată pe grupe (secțiunea 4). Încadrarea este o etichetă de analiză și nu modifică împărțirea setului de date în train, val și test-dev.

### 3.2 Îmbunătățirea contrastului

Îmbunătățirea contrastului folosește CLAHE (Contrast Limited Adaptive Histogram Equalization; Zuiderveld, 1994). Spre deosebire de egalizarea globală a histogramei, CLAHE împarte imaginea într-o grilă de regiuni și egalizează histograma fiecărei regiuni separat. Valorile din histogramă care depășesc o limită (clip limit) sunt tăiate și redistribuite uniform, ca să nu fie amplificat zgomotul în zonele omogene. La final, rezultatele regiunilor vecine sunt îmbinate prin interpolare biliniară, ca să nu apară granițe vizibile între ele.

CLAHE se aplică doar pe canalul de luminanță (de exemplu canalul L din spațiul Lab), iar culorile rămân neschimbate, pentru că detectorul folosește și informația de culoare. Sunt studiate două variante. Varianta uniformă aplică CLAHE tuturor imaginilor. Varianta adaptivă îl aplică doar imaginilor din grupele întuneric și contrast slab, iar imaginile normale trec neatinse. Parametrii CLAHE nu sunt optimizați, pentru că fiecare încercare ar cere o reantrenare a modelului. Se folosesc valori uzuale (o limită de tăiere de 2 și o grilă de 8×8 regiuni), menționate explicit în raport.

Îmbunătățirea se aplică întotdeauna pe imaginea întreagă, înainte de orice împărțire în ferestre, și identic la antrenare și la testare. Aplicarea la nivel de imagine are trei motive. CLAHE este deja o operație locală, care se adaptează zonelor diferite din interiorul imaginii. Indicatorii statistici calculați pe o fereastră mică sunt instabili. Iar grupele trebuie definite la nivel de imagine pentru ca evaluarea pe grupe să fie posibilă.

### 3.3 Felierea pe grilă fixă

Felierea împarte imaginea în ferestre pătrate de aceeași dimensiune cu intrarea detectorului, așezate pe rânduri și coloane. Ferestrele vecine se suprapun parțial, astfel încât un obiect tăiat de marginea unei ferestre apare întreg în fereastra vecină. Pozițiile ferestrelor sunt alese astfel încât să acopere toată imaginea. Pe lângă ferestre, detectorul primește și imaginea întreagă, micșorată, care asigură detecția obiectelor mari, ce nu încap într-o singură fereastră. În implementarea curentă, dimensiunea ferestrei și gradul de suprapunere sunt parametri de configurare.

Ca alternativă la această implementare este evaluată biblioteca SAHI (Slicing Aided Hyper Inference; Akyon et al., 2022), care realizează aceeași idee de feliere la inferență. Cele două variante nu se folosesc împreună în aceeași rulare: sunt două realizări ale aceluiași pas, comparate între ele.

### 3.4 Detecția preliminară și ferestrele adaptive (planificat)

Grila fixă analizează fiecare zonă a imaginii, inclusiv zonele fără obiecte (câmpuri, acoperișuri, cer), ceea ce costă timp. Componenta planificată înlocuiește grila cu ferestre plasate doar acolo unde există obiecte, în doi pași.

Primul pas, detecția preliminară, rulează detectorul pe imaginea micșorată și obține o estimare aproximativă a pozițiilor obiectelor. Ideea de a căuta întâi grosier și de a analiza detaliat doar zonele promițătoare este preluată din QueryDet (Yang et al., 2022). Acolo, căutarea grosieră se face însă în interiorul rețelei neuronale, pe hărțile de caracteristici de rezoluție mică, și este antrenată împreună cu detectorul.

Al doilea pas transformă detecțiile preliminare în ferestre prin operații clasice de prelucrare a imaginilor. Detecțiile sunt desenate ca regiuni albe pe o imagine binară neagră (mască). Masca este dilatată morfologic, astfel încât regiunile apropiate să se unească. Fiecare componentă conexă rămasă reprezintă un grup de obiecte, iar dreptunghiul care o încadrează devine o fereastră de analiză. Ideea de a grupa obiectele în regiuni și de a analiza separat fiecare regiune vine din ClusDet (Yang et al., 2019), unde regiunile sunt propuse de o rețea separată, antrenată în acest scop. O abordare apropiată, bazată tot pe detecții inițiale, este Cascaded Zoom-in (Meethal et al., 2023). Contribuția proiectului nu este ideea în sine, ci realizarea ei fără nicio rețea suplimentară, numai cu operații clasice: binarizare, dilatare și etichetarea componentelor conexe.

Metoda are două limite cunoscute. Obiectele pe care detecția preliminară nu le vede deloc pe imaginea micșorată nu primesc fereastră. Iar gradul de dilatare trebuie reglat: o dilatare prea mică produce multe ferestre mici, iar una prea mare produce grupuri foarte extinse, care trebuie la rândul lor împărțite. Pentru prima limită este prevăzută varianta în care ferestrele adaptive se adaugă grilei, care rămâne ca plasă de siguranță.

### 3.5 Detecția pe ferestre

Detectorul antrenat rulează pe fiecare fereastră, indiferent de modul în care au fost alese ferestrele. Chenarele obținute sunt exprimate în coordonatele ferestrei, așa că sunt mutate în coordonatele imaginii întregi prin adunarea poziției colțului stânga-sus al ferestrei. De exemplu, un chenar găsit între punctele (100, 50) și (140, 90) într-o fereastră care începe la (512, 0) devine, în imaginea întreagă, chenarul dintre (612, 50) și (652, 90).

### 3.6 Fuziunea detecțiilor

Din cauza suprapunerii dintre ferestre, același obiect poate fi detectat de mai multe ori, iar un obiect tăiat de marginea unei ferestre poate fi detectat ca obiect parțial. Fuziunea rezolvă ambele situații în doi pași.

În primul pas sunt eliminate detecțiile tăiate. O detecție este eliminată dacă atinge, la o distanță de cel mult doi pixeli, o margine a ferestrei aflată în interiorul imaginii. Marginile care coincid cu marginea imaginii nu sunt luate în considerare, pentru că acolo obiectul este tăiat de cadrul fotografiei, nu de fereastră. Obiectul eliminat astfel apare întreg în fereastra vecină, datorită suprapunerii.

În al doilea pas sunt eliminate dublurile, printr-o variantă de suprimare a non-maximelor (NMS). Detecțiile sunt ordonate descrescător după scorul de încredere. O detecție este eliminată dacă se suprapune puternic cu o detecție de aceeași clasă și cu scor mai mare. Măsura de suprapunere folosită nu este IoU (Intersection over Union), ci IOS (Intersection over Smaller):

IoU(A, B) = aria(A ∩ B) / aria(A ∪ B)

IOS(A, B) = aria(A ∩ B) / min(aria(A), aria(B))

Diferența contează când un chenar mic este cuprins aproape în întregime într-unul mare, de exemplu o detecție parțială a unei mașini în interiorul detecției întregi. IoU are în acest caz o valoare mică, iar dublura ar fi păstrată. IOS are o valoare apropiată de 1, iar dublura este eliminată. Pragul implicit de suprimare este 0,5.

### 3.7 Antrenarea modelului

Modelul pornește de la greutățile YOLO26n preantrenate și este adaptat prin fine-tuning pe subsetul de antrenare. Sunt pregătite două configurații de antrenare: una pe imagini întregi și una pe ferestre suprapuse tăiate din imaginile de antrenare. Atunci când se folosește îmbunătățirea contrastului, ea este aplicată imaginilor de antrenare înainte de tăierea ferestrelor, exact ca la testare, ca modelul să fie antrenat pe același tip de imagine pe care îl va primi la utilizare.

Antrenarea nu face parte din drumul unei imagini prin sistem. Ea are loc o singură dată, înainte de orice testare, iar rezultatul ei, fișierul de greutăți, este folosit de componentele de detecție.

## 4. Evaluarea

Detecțiile finale sunt comparate cu adnotările manuale ale imaginii întregi, după reasamblarea și fuziunea detecțiilor din ferestre. Evaluarea pe ferestre separate ar produce valori care nu pot fi comparate cu cele obținute pe imagini întregi.

O detecție este considerată corectă dacă are aceeași clasă ca un obiect adnotat și IoU-ul cu chenarul acestuia depășește un prag. Pentru fiecare clasă se calculează precizia medie (AP), adică aria de sub curba precizie–recall obținută prin varierea pragului de încredere. Media AP pe clase este mAP. Sunt raportate două variante uzuale: mAP50, cu pragul IoU de 0,5, și mAP50-95, media valorilor mAP pentru praguri IoU de la 0,5 la 0,95 cu pasul 0,05. Aceasta din urmă recompensează și precizia localizării.

Pentru ca rulările să fie comparabile, toate sunt evaluate în aceleași condiții. Se folosesc aceleași imagini, același prag de încredere și aceleași praguri de fuziune, iar adnotările de referință sunt cele ale imaginii întregi. Pragul de încredere trebuie să fie același și scăzut pentru toate variantele, pentru că mAP depinde de detecțiile cu scor mic. Un prag mai mare la una dintre variante ar dezavantaja-o artificial.

Pe lângă scorul pe întregul set, mAP este calculat separat pe fiecare grupă de imagini stabilită în secțiunea 3.1. Astfel se poate verifica dacă o metodă ajută în special imaginile întunecate sau slab contrastate, chiar dacă efectul pe întregul set este mic. Este măsurat și timpul mediu de procesare pe imagine, pe același echipament, pentru că felierea crește numărul de rulări ale detectorului. Ca extensie, poate fi raportată și precizia pe categorii de mărime a obiectelor, după convenția COCO, care ar susține direct afirmațiile despre obiectele mici.

## 5. Metodologia experimentală

Experimentele urmează schema din figura de mai jos. Ele sunt organizate în etape: la fiecare etapă se adaugă o singură componentă, iar ea este păstrată doar dacă îmbunătățește rezultatul pe setul de validare. Astfel, fiecare comparație diferă printr-un singur element, iar diferența de rezultat poate fi atribuită acelui element.



În etapa 1 se stabilește modul de împărțire a imaginii. Sunt comparate trei variante. Baseline-ul folosește modelul antrenat pe imagini întregi, testat pe imaginea întreagă, și este referința tuturor comparațiilor. Varianta SAHI folosește același model, testat pe ferestre prin biblioteca SAHI, fără reantrenare. Varianta cu feliere pe grilă fixă folosește modelul antrenat pe ferestre, testat pe ferestre, cu fuziunea descrisă în secțiunea 3.6. Comparația dintre baseline și SAHI arată efectul felierii la testare. Comparația dintre SAHI și felierea pe grilă arată efectul suplimentar al antrenării pe ferestre. Varianta cu cel mai bun mAP pe setul de validare devine configurația selectată.

În etapa 2 se evaluează îmbunătățirea contrastului, pe configurația selectată în etapa 1. Pentru fiecare dintre cele două variante, uniformă și adaptivă, CLAHE este aplicat imaginilor de antrenare înainte de tăiere, modelul este reantrenat, iar testarea pe setul de validare se face în același mod ca în configurația selectată. CLAHE este păstrat doar dacă îmbunătățește rezultatul. Un rezultat negativ, care ar arăta că îmbunătățirea contrastului nu ajută pe acest set de date, este și el o concluzie valabilă și este raportat ca atare.

Etapa 3 este planificată și evaluează ferestrele adaptive pe configurația selectată în etapa 2, în două variante: în locul grilei fixe și împreună cu ea.

După ce toate deciziile au fost luate pe setul de validare, toate variantele sunt rulate o singură dată pe test-dev. Pentru că această rulare nu cere reantrenare, costul ei este doar timpul de inferență. Tabelul final conține, pentru fiecare variantă, valorile pe setul de validare, care justifică alegerile, și valorile pe test-dev, care sunt rezultatele raportate.

| Etapa | Variantă | Antrenare | Testare |
|---|---|---|---|
| 1 | Baseline | imagini întregi | imagine întreagă |
| 1 | SAHI | imagini întregi | ferestre (biblioteca SAHI) |
| 1 | Feliere pe grilă fixă | ferestre | ferestre și imaginea întreagă micșorată |
| 2 | CLAHE uniform | ca în configurația selectată, cu CLAHE pe toate imaginile | la fel |
| 2 | CLAHE adaptiv | ca în configurația selectată, cu CLAHE pe grupele care îl cer | la fel |
| 3 (planificat) | Ferestre adaptive | ca în configurația selectată | ferestre adaptive |
| 3 (planificat) | Ferestre adaptive și grilă | ca în configurația selectată | ferestre adaptive și grilă |

Metodologia are o limită asumată. Selecția pe etape nu verifică toate combinațiile posibile. De exemplu, nu arată dacă CLAHE ar fi avut un efect mai mare pe o variantă de împărțire care nu a fost selectată în etapa 1. Testarea tuturor combinațiilor ar multiplica numărul de antrenări și rămâne o direcție pentru continuarea proiectului.

## 6. Starea implementării

Sunt implementate pregătirea datelor, configurațiile de antrenare pe imagini întregi și pe ferestre, tăierea ferestrelor pentru antrenare și inferența pe ferestre cu fuziunea detecțiilor (eliminarea detecțiilor tăiate și suprimarea dublurilor prin IOS). Urmează analiza statistică a imaginilor, îmbunătățirea contrastului în cele două variante, integrarea bibliotecii SAHI și evaluarea prin mAP a variantelor cu feliere, în aceleași condiții ca baseline-ul. Detecția preliminară și ferestrele adaptive sunt planificate pentru etapa de după prototip.

## Bibliografie

Akyon, F. C., Altinuc, S. O., Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. IEEE International Conference on Image Processing (ICIP).

Meethal, A., Granger, E., Pedersoli, M. (2023). Cascaded Zoom-in Detector for High Resolution Aerial Images. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). arXiv:2303.08747.

Yang, C., Huang, Z., Wang, N. (2022). QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yang, F., Fan, H., Chu, P., Blasch, E., Ling, H. (2019). Clustered Object Detection in Aerial Images. IEEE/CVF International Conference on Computer Vision (ICCV).

Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., Ling, H. (2022). Detection and Tracking Meet Drones Challenge. IEEE Transactions on Pattern Analysis and Machine Intelligence. arXiv:2001.06303.

Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. În P. Heckbert (ed.), Graphics Gems IV, Academic Press, 474–485.

Ultralytics (2026). Ultralytics YOLO, documentație și cod sursă. https://docs.ultralytics.com
