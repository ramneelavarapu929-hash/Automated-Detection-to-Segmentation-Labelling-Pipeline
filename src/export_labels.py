import cv2
import json
from pathlib import Path
from converters.mask_to_yolo_seg import mask_to_yolo
from converters.mask_to_coco import mask_to_coco_segmentation

import cv2, json
from pathlib import Path
from converters.mask_to_yolo_seg import mask_to_yolo

def merge_labels(mask_dir, out_label_dir):
    mask_dir = Path(mask_dir)
    out_label_dir = Path(out_label_dir)
    out_label_dir.mkdir(exist_ok=True)

    groups = {}
    for mask_path in mask_dir.glob("*.png"):
        stem = mask_path.stem.rsplit("_", 1)[0]
        groups.setdefault(stem, []).append(mask_path)

    for stem, masks in groups.items():
        lines = []

        for mask_path in masks:
            mask = cv2.imread(str(mask_path), 0)

            with open(mask_path.with_suffix(".json")) as f:
                meta = json.load(f)
            class_id = meta["class"]

            lines.extend(mask_to_yolo(mask, class_id))

        with open(out_label_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))

        #print(f"[✓] {stem}: {len(lines)} segments")

def export_yolo(mask_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in mask_dir.glob("*.png"):
        mask = cv2.imread(str(mask_path), 0)
        with open(mask_path.with_suffix(".json")) as f:
            meta = json.load(f)
        class_id = meta["class"]
        labels = mask_to_yolo(mask, class_id)

        with open(out_dir / f"{mask_path.stem}.txt", "w") as f:
            f.write("\n".join(labels))

def export_coco(mask_dir, out_json):
    annotations = []
    ann_id = 1

    for mask_path in mask_dir.glob("*.png"):
        mask = cv2.imread(str(mask_path), 0)
        segs = mask_to_coco_segmentation(mask)

        for seg in segs:
            annotations.append({
                "id": ann_id,
                "image_id": mask_path.stem,
                "category_id": 1,
                "segmentation": [seg],
                "iscrowd": 0
            })
            ann_id += 1

    with open(out_json, "w") as f:
        json.dump({"annotations": annotations}, f)

if __name__ == "__main__":
    #export_yolo(Path("data/processed/masks/train/silver"), Path("data/processed/masks/train/silver/labels_sep"))
    #export_coco(Path("data/processed/masks/train/silver"), "data/annotations/coco/annotations.json")
    merge_labels(
        "data/processed/masks/train/review_queue",
        "data/processed/masks/train/review_queue/labels"
    )
