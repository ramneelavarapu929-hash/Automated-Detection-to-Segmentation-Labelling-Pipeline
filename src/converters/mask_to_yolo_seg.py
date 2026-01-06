import numpy as np
import cv2

def mask_to_yolo(mask, class_id):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = mask.shape

    labels = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt.squeeze()
        cnt = cnt.astype(float)
        cnt[:, 0] /= w
        cnt[:, 1] /= h
        labels.append(
            f"{class_id} " + " ".join(map(str, cnt.flatten()))
        )
    return labels
