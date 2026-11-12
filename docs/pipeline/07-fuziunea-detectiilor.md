# 7. Fuziunea detectiilor

**Cod:** `drop_cut`, `to_image` si `merge` din `src/pi_p/predict.py` · folosite de grila si de ferestrele adaptive

## Problema

Ferestrele se suprapun, iar modelul vede si imaginea intreaga. De aceea, lista bruta de detectii are doua feluri de greseli:

1. **Bucati de obiecte.** O masina taiata de marginea unei ferestre e detectata ca „jumatate de masina”, cu un chenar prea mic.
2. **Dubluri.** O masina din zona comuna a doua ferestre e detectata in amandoua, plus o data in imaginea intreaga.

Fuziunea le rezolva in doi pasi, in ordinea asta.

```mermaid
flowchart LR
    A["Detectii brute
din toate ferestrele"] --> B["Pasul 1: drop_cut
arunca detectiile lipite de marginea ferestrei"]
    B --> C["Pasul 2: merge
din fiecare grup de dubluri ramane cea mai sigura"]
    C --> D["Detectii finale"]
```

## Pasul 1: eliminarea detectiilor taiate (`drop_cut`)

O detectie este aruncata daca atinge, la cel mult 2 px, o margine a ferestrei aflata **in interiorul imaginii**.

```
        imaginea intreaga
   ┌──────────────────────────────┐
   │  fereastra A     ┆           │
   │            ┌─────┆──┐        │   masina e taiata de marginea dreapta a lui A:
   │            │ masina │        │   in A apare doar partea stanga → aruncata din A
   │            └─────┆──┘        │   in fereastra B, vecina, apare intreaga → pastrata
   │                  ┆           │
   └──────────────────────────────┘
```

De ce e sigur sa o aruncam: datorita suprapunerii, obiectul apare intreg in fereastra vecina.

Exceptia: marginile ferestrei care coincid cu marginea imaginii. Acolo obiectul e taiat de fotografie, nu de fereastra, si nicio alta fereastra nu il vede mai bine. In cod, fiecare dintre cele patru margini e verificata doar daca fereastra nu e lipita de marginea respectiva a imaginii (`if x1 > 0`, `if x2 < w` etc.).

## Pasul 2: eliminarea dublurilor (`merge`)

Este o suprimare a non-maximelor (NMS):

1. ordonam detectiile descrescator dupa scor;
2. luam prima detectie si o pastram;
3. aruncam toate detectiile de aceeasi clasa care se suprapun puternic cu ea;
4. repetam cu urmatoarea detectie ramasa.

### De ce IOS si nu IoU

Suprapunerea a doua chenare se masoara de obicei cu IoU:

```
IoU(A, B) = aria(A ∩ B) / aria(A ∪ B)
IOS(A, B) = aria(A ∩ B) / min(aria(A), aria(B))
```

IoU are o problema cu bucatile care au scapat de pasul 1. Exemplu: masina intreaga are chenarul de 100 × 50 px, iar o bucata detectata din ea are 40 × 50 px si e complet in interior.

| Masura | Calcul | Valoare | La pragul 0,5 |
|---|---|---|---|
| IoU | 2000 / 5000 | 0,40 | bucata e **pastrata** (gresit) |
| IOS | 2000 / 2000 | 1,00 | bucata e **eliminata** (corect) |

IOS (Intersection over Smaller) imparte la aria chenarului mai mic, deci spune „cat din chenarul mic e acoperit de cel mare”. Un chenar aflat aproape complet in interiorul altuia de aceeasi clasa este aproape sigur o bucata din acelasi obiect.

Pragul implicit este 0,5 si se poate schimba cu `--iou`.

### De ce doar intre detectii de aceeasi clasa

Un pieton urcat pe o motocicleta are chenarul aproape complet in interiorul chenarului motocicletei. Sunt doua obiecte reale, de clase diferite, si trebuie pastrate amandoua.

## Un exemplu complet

Testul `test_merge_drops_fragments_inside_larger_box` din `tests/test_slicing.py`:

| Detectie | Chenar | Scor | Clasa | Rezultat |
|---|---|---|---|---|
| 1 | (0, 0) – (100, 50) | 0,9 | truck | pastrata: cel mai mare scor |
| 2 | (0, 0) – (40, 50) | 0,6 | truck | eliminata: e in interiorul lui 1, aceeasi clasa |
| 3 | (10, 10) – (30, 30) | 0,8 | pedestrian | pastrata: e in interiorul lui 1, dar alta clasa |
| 4 | (200, 200) – (220, 220) | 0,7 | truck | pastrata: nu se suprapune cu nimic |
