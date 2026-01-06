import cv2
import numpy as np

def mask_to_coco_segmentation(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segmentations = []
    for cnt in contours:
        if cnt.shape[0] >= 3:
            segmentations.append(cnt.flatten().tolist())

    return segmentations
