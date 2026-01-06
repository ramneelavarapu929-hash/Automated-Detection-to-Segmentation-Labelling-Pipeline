import cv2
import numpy as np

def refine_mask(mask, min_area=300):
    """
    Morphological cleanup + hole filling
    """
    mask = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Remove small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    refined = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined[labels == i] = 1

    return refined
