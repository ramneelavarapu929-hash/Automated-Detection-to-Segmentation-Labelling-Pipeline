import cv2
import numpy as np
from pathlib import Path

def validate_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    area = np.sum(mask > 0)
    return area

def main(mask_dir):
    mask_dir = Path(mask_dir)
    total, empty = 0, 0

    for mask_path in mask_dir.rglob("*.png"):
        total += 1
        if validate_mask(mask_path) < 50:
            empty += 1

    print("===== Label Validation Report =====")
    print(f"Total masks: {total}")
    print(f"Empty / invalid masks: {empty}")
    print(f"Validity rate: {(1 - empty / max(total,1)) * 100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", default="data/processed/masks")
    args = parser.parse_args()

    main(args.mask_dir)
