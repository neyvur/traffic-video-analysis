import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    """YOLO-based detector for traffic analysis.

    Returns detections of real COCO classes relevant for traffic:
    vehicles, pedestrians, traffic lights, and a few auxiliary classes.
    
    Event classes like 'accident', 'red_light', 'jaywalking' are NOT
    detected here — they are inferred by RulesEngine / EventsDetector
    from tracked objects.
    """

    # Real COCO classes (present in YOLOv8 pretrained weights)
    COCO_TRAFFIC_CLASSES = {
        'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'person', 'traffic light', 'stop sign',
        'fire hydrant',   # may act as "obstacle_on_road"
    }

    def __init__(self, weight_path='weights/model.pt',
                 conf_thres=0.25, iou_thres=0.45,
                 device=None, half=False):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half

        self.model = YOLO(weight_path)
        self.class_names = self.model.names  # dict: {id: name}

        # If the model was custom-trained (fire, smoke, etc.), add those too
        custom_classes = set()
        for name in self.class_names.values():
            n = name.lower()
            if n in {'fire', 'smoke', 'accident', 'near_miss'}:
                custom_classes.add(name)

        self.target_classes = self.COCO_TRAFFIC_CLASSES | custom_classes

    # ------------------------------------------------------------------

    def detect(self, frame):
        """Detect objects in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            list of dicts:
              {'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float}
        """
        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
            class_name = self.class_names.get(cid, str(cid))
            if class_name not in self.target_classes:
                continue
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name,
                'confidence': float(conf),
            })

        return detections
    