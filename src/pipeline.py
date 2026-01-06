import cv2
from pathlib import Path
from .detection.yolo_infer import YOLOPromptGenerator
from .segmentation.sam2_image import SAM2ImageSegmenter
from .segmentation.mask_refinement import refine_mask
from .hil.uncertainty import compute_uncertainty
import json

class DetectionToSegmentationPipeline:
    def __init__(self, cfg, uncertainty_threshold=0.5):
        self.detector = YOLOPromptGenerator(
            cfg["yolo_weights"],
            conf=cfg.get("conf", 0.25)
        )
        self.segmenter = SAM2ImageSegmenter(
            model_cfg=cfg["sam_model_cfg"],
            checkpoint=cfg["sam_ckpt"],
            device=cfg["device"]
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.class_rarity = cfg.get("class_rarity", {})
    
    def process_directory(self, img_dir, out_root):
        img_dir = Path(img_dir)
        out_root = Path(out_root)

        silver_dir = out_root / "silver"
        review_dir = out_root / "review_queue"

        silver_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.glob("*.jpg"):
            self.process_image(img_path, silver_dir, review_dir)

    def process_image(self, img_path, silver_dir, review_dir):
        image = cv2.imread(str(img_path))
        if image is None:
            return

        boxes, classes, scores = self.detector.detect(image)
        masks = self.segmenter.segment(image, boxes)

        for idx, (box, cls, conf, mask) in enumerate(
            zip(boxes, classes, scores, masks)
        ):
            refined = refine_mask(mask)

            rarity = self.class_rarity.get(int(cls), 0.0)
            uncertainty = compute_uncertainty(
                conf, box, refined, rarity
            )

            record = {
                "image": str(img_path),
                "box": box.tolist(),
                "class": int(cls),
                "confidence": float(conf),
                "uncertainty": float(uncertainty)
            }

            if uncertainty <= self.uncertainty_threshold:
                self._save_instance(
                    refined, record, silver_dir, img_path, idx
                )
            else:
                self._save_instance(
                    refined, record, review_dir, img_path, idx
                )

    def _save_instance(self, mask, record, out_dir, img_path, idx):
        stem = f"{img_path.stem}_{idx}"
        cv2.imwrite(
            str(out_dir / f"{stem}.png"),
            mask * 255
        )

        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(record, f, indent=2)
