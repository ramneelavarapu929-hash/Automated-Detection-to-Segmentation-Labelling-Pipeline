import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2ImageSegmenter:
    """
    Segment static images using SAM 2.1 with YOLO bounding boxes as prompts.
    """

    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cuda"):
        """
        Args:
            model_cfg (str): Path to the SAM2 YAML config file.
            checkpoint (str): Path to the SAM2 checkpoint (*.pt).
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device

        # Build and load the SAM2 model
        self.sam_model = build_sam2(model_cfg, checkpoint)
        self.predictor = SAM2ImagePredictor(self.sam_model)

        # Move to correct device
        if device == "cuda":
            self.predictor.model.to(device)
        
        self.predictor.model.eval()

    def segment(self, image: np.ndarray, boxes: np.ndarray):
        """
        Segment an image given YOLO bounding boxes.

        Args:
            image (np.ndarray): HWC image (BGR, OpenCV format).
            boxes (np.ndarray): Nx4 array of bounding boxes [x1, y1, x2, y2].

        Returns:
            List[np.ndarray]: Binary masks (same HxW as the input image).
        """
        # Convert BGR -> RGB
        image_rgb = image[:, :, ::-1].copy()

        with torch.inference_mode():
            # Set the image once
            self.predictor.set_image(image_rgb)

            masks = []
            for box in boxes:
                # SAM2 expects prompt format = box
                try:
                    out_masks, _, _ = self.predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    # out_masks is (N_masks, H, W)
                    masks.append(out_masks[0].astype(np.uint8))
                except Exception as e:
                    print(f"[!] SAM2 failed on box {box}: {e}")
                    masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))

        return masks