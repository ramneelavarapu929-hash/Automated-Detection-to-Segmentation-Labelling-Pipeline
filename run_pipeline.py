import argparse
import yaml
from pathlib import Path
from src.pipeline import DetectionToSegmentationPipeline

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    yolo_cfg = load_yaml(args.yolo_cfg)
    sam_cfg = load_yaml(args.sam_cfg)

    cfg = {
        "yolo_weights": yolo_cfg["model"]["weights"],
        "sam_model_cfg": sam_cfg["model"]["cfg"],
        "sam_ckpt": sam_cfg["model"]["checkpoint"],
        "device": sam_cfg["model"]["device"],
        "class_rarity": sam_cfg["class_rarity"]
    }

    pipeline = DetectionToSegmentationPipeline(cfg)

    image_root = Path(args.image_root)
    out_root = Path(args.out_root)

    for split in ["train", "val", "test"]:
        img_dir = image_root / split
        out_dir = out_root / split

        if not img_dir.exists():
            continue

        print(f"[→] Processing {split} images")
        pipeline.process_directory(img_dir, out_dir)

    print("[✓] Image-based Detection-to-Segmentation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="data/raw/images")
    parser.add_argument("--out_root", default="data/processed/masks")
    parser.add_argument("--yolo_cfg", default="configs/yolo.yaml")
    parser.add_argument("--sam_cfg", default="configs/sam.yaml")
    args = parser.parse_args()

    main(args)
