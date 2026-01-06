from ultralytics import YOLO

class YOLOPromptGenerator:
    def __init__(self, weights, conf=0.25):
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image):
        results = self.model(image, conf=self.conf, classes=[2,3,5,6,7], verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        return boxes, classes, scores