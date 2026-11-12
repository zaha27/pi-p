# Demo

Doua video de 3 minute, facute din 90 de imagini din VisDrone test-dev (imagini pe care modelul nu le-a vazut la antrenare), cate 2 secunde fiecare.

| Fisier | Continut |
|---|---|
| `demo_visdrone_test_detectii.mp4` | detectiile modelului `weights/yolo26n_tiled.pt` cu grila de 640 px, desenate pe imagini |
| `demo_visdrone_test.mp4` | aceleasi imagini, fara detectii, pentru rularea live in aplicatie (se copiaza in `videos/`) |

Imaginile sunt separate, nu un film continuu, deci urmarirea si viteza din dashboard nu au sens pe aceste video.
