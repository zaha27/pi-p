# Ponderi antrenate

Modele YOLO26n fine-tuned pe VisDrone2019-DET, antrenate pe un RTX 4070 SUPER. Fiecare fișier este `best.pt` din antrenarea respectivă.

| Fișier | Configurare | Date de antrenare | mAP50 pe val (`pi-eval`) |
|---|---|---|---|
| `yolo26n_baseline_640.pt` | `configs/train.yaml` | 6471 imagini întregi | 0,307 (imagine întreagă), 0,392 (grilă) |
| `yolo26n_tiled.pt` | `configs/train_tiled.yaml` | felii de 640 px + imagini întregi | 0,327 (imagine întreagă), 0,483 (grilă) |
| `yolo26n_tiled_clahe_uniform.pt` | `configs/train_tiled.yaml`, date cu CLAHE uniform | felii, după CLAHE pe toate imaginile | 0,325 (imagine întreagă), 0,482 (grilă) |
| `yolo26n_tiled_clahe_adaptiv.pt` | `configs/train_tiled.yaml`, date cu CLAHE adaptiv | felii, după CLAHE doar pe imaginile întunecate sau slab contrastate | 0,326 (imagine întreagă), 0,484 (grilă) |

Valorile sunt măsurate pe imaginile întregi din val, cu prag de încredere 0,001. Modelele CLAHE se evaluează pe setul de date cu același CLAHE (`--data datasets/VisDrone-clahe-.../data.yaml`).

Folosire:

```bash
uv run pi-predict weights/yolo26n_tiled.pt imagine.jpg --slice 640
```
