import numpy as np
from collections import defaultdict


class ByteTrack:
    """Обёртка над встроенным ByteTrack Ultralytics.

    Модель YOLO сама делает детекцию + трекинг через model.track().
    """

    # Классы COCO, которые нам интересны
    TRAFFIC_CLASSES = {
        'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'person', 'traffic light', 'stop sign',
    }

    def __init__(self, yolo, frame_rate=30, conf_thres=0.25,
                 iou_thres=0.45, track_buffer=60, match_thresh=0.8):
        """
        Args:
            yolo: загруженная модель Ultralytics YOLO
            frame_rate: FPS видео
            conf_thres: порог confidence детекции
            iou_thres: порог IoU для NMS
            track_buffer: сколько кадров держать потерянный трек
            match_thresh: порог сопоставления ByteTrack
        """
        self.yolo = yolo
        self.frame_rate = frame_rate
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Настройки трекинга
        self.tracker_cfg = 'bytetrack.yaml'  # или 'botsort.yaml'

        # Статистика
        self.trajectory_data = defaultdict(lambda: {'positions': [], 'classes': []})
        self.frame_count = 0

    def update(self, frame):
        """Обрабатывает кадр, возвращает список треков.

        Args:
            frame: BGR-кадр

        Returns:
            list of {'track_id': int, 'bbox': [x1,y1,x2,y2],
                     'class': str, 'confidence': float}
        """
        self.frame_count += 1

        results = self.yolo.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        if not results:
            return []

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        tracks = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # track_id может быть None (если объект новый и трек ещё не присвоен)
        ids = boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            ids = [None] * len(xyxy)

        for (x1, y1, x2, y2), conf, cid, tid in zip(xyxy, confs, clss, ids):
            class_name = self.yolo.names.get(cid, str(cid))
            if class_name not in self.TRAFFIC_CLASSES:
                continue
            if tid is None:
                continue  # пропускаем треки без ID (первый кадр появления)

            track = {
                'track_id': int(tid),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name,
                'confidence': float(conf),
            }
            tracks.append(track)

            # Храним историю позиций
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.trajectory_data[int(tid)]['positions'].append([cx, cy])
            self.trajectory_data[int(tid)]['classes'].append(class_name)
            if len(self.trajectory_data[int(tid)]['positions']) > 120:
                self.trajectory_data[int(tid)]['positions'] = \
                    self.trajectory_data[int(tid)]['positions'][-120:]
                self.trajectory_data[int(tid)]['classes'] = \
                    self.trajectory_data[int(tid)]['classes'][-120:]

        return tracks

    def get_trajectory_data(self):
        return dict(self.trajectory_data)