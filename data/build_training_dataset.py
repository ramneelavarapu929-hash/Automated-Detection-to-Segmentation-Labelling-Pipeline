import shutil
import random
from pathlib import Path

def copy_subset(src_img, src_lbl, dst_img, dst_lbl, ratio=1.0):
    print(src_img,src_lbl)
    imgs = list(src_img.glob("*.jpg"))
    print(len(imgs))
    random.shuffle(imgs)
    imgs = imgs[:int(len(imgs) * ratio)]

    for img in imgs:
        lbl = src_lbl / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(img, dst_img / img.name)
            shutil.copy(lbl, dst_lbl / lbl.name)

def main():
    base = Path("data/")
    out = Path("data/final_dataset")

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Silver partially included
    copy_subset(
        base / "raw/images/train",
        base / "processed/masks/train/silver/labels",
        out / "images/train",
        out / "labels/train",
        ratio=0.8
    )

    # Validation & test are gold-only
    copy_subset(
        base / "raw/images/train",
        base / "processed/masks/train/silver/labels",
        out / "images/val",
        out / "labels/val",
        ratio=0.2
    )

if __name__ == "__main__":
    main()
