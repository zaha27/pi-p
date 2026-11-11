"""Ferestre adaptive: plasează ferestre doar acolo unde sunt obiecte.

Grila fixă (slicing.py) analizează toată imaginea, inclusiv câmpuri și
acoperișuri goale. Aici alegem ferestrele pornind de la o detecție
preliminară, făcută pe imaginea micșorată:

  1. desenăm chenarele detecției preliminare pe o mască alb-negru;
  2. dilatăm masca, ca obiectele apropiate să se unească într-o pată;
  3. fiecare pată (componentă conexă) este un grup de obiecte;
  4. peste fiecare grup punem una sau mai multe ferestre de mărime fixă.

Nu se antrenează nimic în plus: sunt doar operații clasice de prelucrare a
imaginilor. Ideea de a căuta întâi grosier vine din QueryDet, iar gruparea
în regiuni din ClusDet. Detalii în docs/pipeline/09-ferestre-adaptive.md.
"""

import cv2
import numpy as np

from pi_p.slicing import Window, windows


def boxes_to_mask(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Desenează chenarele (x1, y1, x2, y2) ca dreptunghiuri albe pe o mască neagră."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.round().astype(int):
        mask[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)] = 255
    return mask


def group_boxes(mask: np.ndarray, dilate: int) -> list[Window]:
    """Unește petele apropiate și întoarce dreptunghiul care încadrează fiecare grup."""
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    groups = []
    # Componenta 0 este fundalul, deci pornim de la 1.
    for i in range(1, count):
        x, y, bw, bh = stats[i, :4]
        groups.append((int(x), int(y), int(x + bw), int(y + bh)))
    return groups


def cover_group(group: Window, w: int, h: int, size: int, overlap: float) -> list[Window]:
    """Acoperă un grup cu ferestre de `size` x `size` pixeli.

    Un grup care încape într-o fereastră primește una singură, centrată pe el.
    Un grup mai mare primește o mică grilă de ferestre suprapuse. Ferestrele
    au mereu aceeași mărime ca la antrenare și sunt împinse înapoi în imagine
    dacă ar ieși din ea.
    """
    gx1, gy1, gx2, gy2 = group
    gw, gh = gx2 - gx1, gy2 - gy1

    # Zona pe care o acoperim: grupul, lărgit până la cel puțin o fereastră.
    zone_w, zone_h = max(gw, size), max(gh, size)
    zone_x = int(np.clip(gx1 - (zone_w - gw) // 2, 0, max(w - zone_w, 0)))
    zone_y = int(np.clip(gy1 - (zone_h - gh) // 2, 0, max(h - zone_h, 0)))
    zone_w, zone_h = min(zone_w, w), min(zone_h, h)

    # Grila din interiorul zonei, mutată în coordonatele imaginii.
    return [(x1 + zone_x, y1 + zone_y, x2 + zone_x, y2 + zone_y) for x1, y1, x2, y2 in windows(zone_w, zone_h, size, overlap)]


def adaptive_windows(boxes: np.ndarray, w: int, h: int, size: int, overlap: float, dilate: int) -> list[Window]:
    """Ferestrele adaptive ale unei imagini de `w` x `h`, pornind de la chenarele preliminare.

    `dilate` este raza dilatării, în pixeli: obiectele aflate la mai puțin de
    aproximativ 2 x `dilate` unul de altul ajung în același grup.
    """
    if len(boxes) == 0:
        return []
    mask = boxes_to_mask(boxes, w, h)
    wins = []
    for group in group_boxes(mask, dilate):
        wins.extend(cover_group(group, w, h, size, overlap))
    # Două grupuri apropiate pot produce aceeași fereastră; o păstrăm o singură dată.
    return sorted(set(wins))
