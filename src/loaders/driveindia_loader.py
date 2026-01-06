import cv2
import os
from pathlib import Path

class DriveIndiaLoader:
    def __init__(self, root, fps=10):
        self.root = Path(root)
        self.video_dir = self.root / "videos"
        self.image_dir = self.root / "images"
        self.fps = fps

    def extract_frames(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for video in self.video_dir.glob("*.mp4"):
            cap = cv2.VideoCapture(str(video))
            frame_id, saved = 0, 0
            stride = int(cap.get(cv2.CAP_PROP_FPS) // self.fps)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % stride == 0:
                    out_path = out_dir / f"{video.stem}_{saved:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved += 1
                frame_id += 1
            cap.release()
