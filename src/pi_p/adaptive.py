import numpy as np

MASK_ON = 255


def boxes_to_mask(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        # floor/ceil acoperă tot obiectul; max/min taie caseta la marginile imaginii
        left, top = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        right, bottom = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = MASK_ON
    return mask
