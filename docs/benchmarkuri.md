```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'16px','lineColor':'#4B5563'}}}%%
flowchart TB
    V["`**Date VisDrone**
train: antrenare; val: comparatii si decizii; test-dev: o singura rulare, la final`"]
    TI["`**Antrenare pe imagini intregi**
fine-tuning YOLO26n pe train, pornind de la modelul preantrenat`"]
    TF["`**Antrenare pe felii**
acelasi fine-tuning, pe felii suprapuse taiate din imaginile de train`"]

    BL["`**Baseline**
modelul pe imagini intregi, testat pe imaginea intreaga; referinta tuturor comparatiilor`"]
    SA["`**SAHI**
acelasi model, testat pe felii prin biblioteca SAHI, fara reantrenare`"]
    GR["`**Feliere pe grila fixa**
modelul pe felii, testat pe felii; detectiile unite prin fuziune`"]
    E1["`**Etapa 1 · evaluare pe setul de validare**
acelasi prag de incredere; mAP total si pe grupe, timp pe imagine; se alege configuratia cu cel mai bun mAP`"]

    CU["`**CLAHE uniform**
CLAHE pe imaginea intreaga, inainte de taiere, la toate imaginile; reantrenare si test ca in configuratia selectata`"]
    CA["`**CLAHE adaptiv**
analiza statistica decide: CLAHE doar pe grupele intuneric si contrast slab; reantrenare si test ca in configuratia selectata`"]
    E2["`**Etapa 2 · evaluare pe setul de validare**
comparatie cu configuratia selectata in etapa 1; CLAHE ramane doar daca imbunatateste rezultatul`"]

    FA["`**Ferestre adaptive**
detectie preliminara (inspirata din QueryDet) si grupare in ferestre (inspirata din ClusDet), in locul grilei`"]
    FG["`**Ferestre adaptive + grila**
grila ramane ca plasa de siguranta pentru obiectele ratate de detectia preliminara`"]
    E3["`**Etapa 3 · evaluare pe setul de validare**
comparatie cu configuratia selectata in etapa 2`"]

    TD["`**Test final pe test-dev**
dupa inghetarea deciziilor, toate variantele rulate o singura data`"]
    TR["`**Tabel de rezultate**
o linie pentru fiecare varianta: valori pe val si pe test-dev`"]

    V --> TI
    V --> TF
    TI --> BL
    TI --> SA
    TF --> GR
    BL --> E1
    SA --> E1
    GR --> E1
    E1 --> CU
    E1 --> CA
    CU --> E2
    CA --> E2
    E2 --> TD
    E2 -.-> FA
    E2 -.-> FG
    FA -.-> E3
    FG -.-> E3
    E3 -.-> TD
    TD --> TR

    LEG["⬜ date &nbsp; 🟩 antrenare &nbsp; 🟦 variante testate &nbsp; 🟨 evaluare &nbsp; 🟪 planificat"]
    TR ~~~ LEG

    classDef dat fill:#F3F4F6,stroke:#374151,stroke-width:2px,color:#111827
    classDef trn fill:#DCFCE7,stroke:#15803D,stroke-width:2px,color:#111827
    classDef var fill:#DBEAFE,stroke:#1D4ED8,stroke-width:2px,color:#111827
    classDef eva fill:#FEF3C7,stroke:#B45309,stroke-width:2px,color:#111827
    classDef plan fill:#F3E8FF,stroke:#7E22CE,stroke-width:2px,stroke-dasharray:6 4,color:#111827
    classDef leg fill:#FFFFFF,stroke:#D1D5DB,stroke-width:1px,color:#374151,font-size:12px

    class V dat
    class TI,TF trn
    class BL,SA,GR,CU,CA var
    class E1,E2,TD,TR eva
    class FA,FG,E3 plan
    class LEG leg
```
