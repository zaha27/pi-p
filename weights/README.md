# Ponderi antrenate

Modele YOLO26n fine-tuned pe VisDrone2019-DET, antrenate pe un RTX 4070 SUPER. Fiecare fisier este `best.pt` din antrenarea respectiva.

| Fisier | Configurare | Date de antrenare | mAP50 pe val (`pi-eval`) |
|---|---|---|---|
| `yolo26n_baseline_640.pt` | `configs/train.yaml` | 6471 imagini intregi | 0,307 (imagine intreaga), 0,392 (grila) |
| `yolo26n_tiled.pt` | `configs/train_tiled.yaml` | felii de 640 px + imagini intregi | 0,327 (imagine intreaga), 0,483 (grila) |
| `yolo26n_tiled_clahe_uniform.pt` | `configs/train_tiled.yaml`, date cu CLAHE uniform | felii, dupa CLAHE pe toate imaginile | 0,325 (imagine intreaga), 0,482 (grila) |
| `yolo26n_tiled_clahe_adaptiv.pt` | `configs/train_tiled.yaml`, date cu CLAHE adaptiv | felii, dupa CLAHE doar pe imaginile intunecate sau slab contrastate | 0,326 (imagine intreaga), 0,484 (grila) |

Valorile sunt masurate pe imaginile intregi din val, cu prag de incredere 0,001. Modelele CLAHE se evalueaza pe setul de date cu acelasi CLAHE (`--data datasets/VisDrone-clahe-.../data.yaml`).

Folosire:

```bash
uv run pi-predict weights/yolo26n_tiled.pt imagine.jpg --slice 640
```
