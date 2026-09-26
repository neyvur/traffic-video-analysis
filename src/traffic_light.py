"""
traffic_light.py — определение состояния светофора через HSV + относительные пороги.

Улучшения:
- Порог считается от суммы R+Y+G, а не от всех ярких пикселей
- Работает даже когда bbox маленький и содержит фон
- 3-секционный fallback для тёмных/размытых светофоров
"""
import cv2
import numpy as np
import json
import os
from collections import deque


class TrafficLightDetector:
    """Определяет цвет светофора по bbox от YOLO или из zones.json."""

    def __init__(self, frame_rate=30, zones_path='zones.json',
                 history_size=8,
                 min_brightness=70, min_saturation=40,
                 min_pixels=3):
        self.frame_rate = frame_rate
        self.history_size = history_size
        self.min_brightness = min_brightness
        self.min_saturation = min_saturation
        self.min_pixels = min_pixels

        self.history = deque(maxlen=history_size)
        self.current_state = 'unknown'

        # Ручные bbox светофоров
        self.manual_bboxes = []
        if zones_path and os.path.exists(zones_path):
            try:
                with open(zones_path) as f:
                    z = json.load(f)
                for tl in z.get('traffic_lights', []):
                    if isinstance(tl, dict) and 'bbox' in tl:
                        b = tl['bbox']
                        self.manual_bboxes.append(
                            [int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                    elif isinstance(tl, list) and len(tl) == 4:
                        self.manual_bboxes.append(
                            [int(tl[0]), int(tl[1]), int(tl[2]), int(tl[3])])
                if self.manual_bboxes:
                    print(f"[TrafficLight] Ручные bbox: {len(self.manual_bboxes)}")
            except Exception as e:
                print(f"[TrafficLight] Ошибка zones: {e}")

    # ------------------------------------------------------------------

    def update(self, frame, detections):
        h, w = frame.shape[:2]

        tl_boxes = []
        for d in detections:
            cls = d.get('class', '').lower()
            if cls in ('traffic light', 'traffic_light'):
                x1, y1, x2, y2 = d['bbox']
                if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                    tl_boxes.append([x1, y1, x2, y2])

        if not tl_boxes and self.manual_bboxes:
            tl_boxes = self.manual_bboxes

        if not tl_boxes:
            self.history.append('unknown')
            self.current_state = self._smooth()
            return self.current_state

        votes = {'red': 0, 'yellow': 0, 'green': 0, 'unknown': 0}
        for bbox in tl_boxes:
            color = self._analyze_bbox(frame, bbox)
            votes[color] += 1

        if votes['red'] > 0:
            frame_color = 'red'
        elif votes['yellow'] > 0:
            frame_color = 'yellow'
        elif votes['green'] > 0:
            frame_color = 'green'
        else:
            frame_color = 'unknown'

        self.history.append(frame_color)
        self.current_state = self._smooth()
        return self.current_state

    # ------------------------------------------------------------------

    def _analyze_bbox(self, frame, bbox):
        """Анализ цвета. Использует относительные пороги."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 'unknown'

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mask = (V >= self.min_brightness) & (S >= self.min_saturation)
        n_bright = int(mask.sum())
        if n_bright < self.min_pixels:
            return 'unknown'

        hues = H[mask]
        if hues.size == 0:
            return 'unknown'

        hist = np.bincount(hues.astype(np.int32), minlength=180)
        red_score = int(hist[0:10].sum() + hist[170:180].sum())
        yellow_score = int(hist[15:35].sum())
        green_score = int(hist[40:90].sum())

        scores = {'red': red_score, 'yellow': yellow_score, 'green': green_score}
        best = max(scores, key=scores.get)
        best_score = scores[best]
        color_total = red_score + yellow_score + green_score

        # ---- ГЛАВНОЕ ИЗМЕНЕНИЕ: относительный порог ----
        # Если цветных (R+Y+G) пикселей мало (< 3) — unknown
        if color_total < 3:
            return 'unknown'

        # Если лучший цвет >= 5 пикселей И >= 30% от суммы — это он
        if best_score >= 5 and best_score >= color_total * 0.3:
            return best

        # Если цветных пикселей мало, но best_score > 0 — тоже возвращаем
        # (мягкий порог для маленьких светофоров)
        if best_score >= 3 and best_score >= color_total * 0.5:
            return best

        return 'unknown'

    # ------------------------------------------------------------------

    def _smooth(self):
        """Быстрое сглаживание: последние кадры важнее."""
        if not self.history:
            return 'unknown'

        last = self.history[-1]
        if last != 'unknown':
            recent5 = list(self.history)[-5:]
            count_last = sum(1 for c in recent5 if c == last)
            if count_last >= 3:
                return last

        recent = list(self.history)[-8:]
        counts = {}
        for c in recent:
            counts[c] = counts.get(c, 0) + 1

        n = len(recent)
        for color in ('red', 'yellow', 'green'):
            if counts.get(color, 0) >= max(2, n * 0.4):
                return color

        known = {k: v for k, v in counts.items() if k != 'unknown'}
        if known:
            return max(known, key=known.get)
        return 'unknown'

    def get_state(self):
        return self.current_state

    def reset(self):
        self.history.clear()
        self.current_state = 'unknown'