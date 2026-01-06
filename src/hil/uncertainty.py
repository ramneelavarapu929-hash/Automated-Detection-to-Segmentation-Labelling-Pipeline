import cv2
import numpy as np

def mask_irregularity(mask):
    area = np.sum(mask)
    if area == 0:
        return 1.0

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0

    perimeter = cv2.arcLength(contours[0], True)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return min(compactness / 10.0, 1.0)

def box_mask_iou(box, mask):
    x1, y1, x2, y2 = map(int, box)
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1

    intersection = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()

    return intersection / max(union, 1)

def compute_uncertainty(yolo_conf, box, mask, class_rarity):
    return (
        0.4 * (1 - yolo_conf) +
        0.3 * mask_irregularity(mask) +
        0.2 * (1 - box_mask_iou(box, mask)) +
        0.1 * class_rarity
    )
