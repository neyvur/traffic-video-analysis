"""
api/annotate.py — прогон пайплайна с записью аннотированного видео.

Функция process_video_annotated():
  - YOLO + ByteTrack
  - RulesEngine + EventsDetector + RiskEstimator
  - Рисует bbox, track_id, события, светофор
  - Пишет в output_path
  - Возвращает (events, risk_scores)
"""
import os
import json
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from src.tracker import ByteTrack
from src.rules import RulesEngine
from src.events import EventsDetector
from src.risk import RiskEstimator
from src.traffic_light import TrafficLightDetector


# ---- Цвета ----
C_OK        = (0, 220, 0)
C_VIOLATION = (0, 0, 255)
C_STOPPED   = (200, 0, 200)
C_PED_IN    = (255, 180, 0)
C_PED_OUT   = (0, 165, 255)
C_UNKNOWN   = (180, 180, 180)

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
PED_CLASSES = {'person', 'pedestrian'}


# =====================================================================
# Геометрия
# =====================================================================

def segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
            ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_in_poly(pt, poly):
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def bbox_intersects_polygon(bbox, poly):
    x1, y1, x2, y2 = bbox
    for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        if point_in_poly((cx, cy), poly):
            return True
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if point_in_poly((cx, cy), poly):
        return True
    poly_pts = [tuple(p) for p in poly]
    n = len(poly_pts)
    bbox_edges = [
        ((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)),
    ]
    for i in range(n):
        p3 = poly_pts[i]
        p4 = poly_pts[(i + 1) % n]
        for p1, p2 in bbox_edges:
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False


# =====================================================================
# Основная функция
# =====================================================================

def process_video_annotated(video_path, output_path, zones_path='zones.json',
                            conf_thres=0.25):
    """Прогон + запись аннотированного видео.

    Returns:
        (events, risk_scores)
        events:      [[start, end, type], ...]
        risk_scores: [(t_sec, risk), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter — H.264 (avc1) для воспроизведения в Chrome/Firefox
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[annotate] avc1 not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Модели
    yolo = YOLO('weights/model.pt')
    tracker = ByteTrack(yolo=yolo, frame_rate=fps, conf_thres=conf_thres)
    risk_estimator = RiskEstimator(frame_rate=fps)
    # Congestion отключён (слишком много ложных на тестовых видео)
    rules = RulesEngine(
        frame_rate=fps,
        zones_path=zones_path,
        risk_estimator=risk_estimator,
        congestion_vehicle_count=999,
        congestion_speed_thresh=0.1,
    )
    events_det = EventsDetector(
        frame_rate=fps,
        risk_estimator=risk_estimator,
        congestion_vehicle_count=999,
        congestion_avg_speed_thresh=0.1,
    )
    tl_det = TrafficLightDetector(frame_rate=fps, zones_path=zones_path)

    # ---- Загрузка и масштабирование зон ----
    # zones.json размечен для REF_W × REF_H. Для другого разрешения — масштабируем.
    REF_W, REF_H = 1280, 720
    sx = w / REF_W
    sy = h / REF_H
    print(f"[zones] video {w}x{h}, scale={sx:.3f}x{sy:.3f}")

    def _scale_pts(pts):
        return [[int(p[0] * sx), int(p[1] * sy)] for p in pts]

    zones = {'stop_lines': [], 'crosswalk_zones': [], 'solid_lines': [],
             'traffic_lights': []}
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            z = json.load(f)

        for line in z.get('stop_lines', []):
            zones['stop_lines'].append((tuple(_scale_pts(line)[0]),
                                        tuple(_scale_pts(line)[1])))
        for line in z.get('solid_lines', []):
            zones['solid_lines'].append((tuple(_scale_pts(line)[0]),
                                         tuple(_scale_pts(line)[1])))
        for poly in z.get('crosswalk_zones', []):
            zones['crosswalk_zones'].append(_scale_pts(poly))
        for tl in z.get('traffic_lights', []):
            if isinstance(tl, dict) and 'bbox' in tl:
                b = tl['bbox']
            elif isinstance(tl, list) and len(tl) == 4:
                b = tl
            else:
                continue
            zones['traffic_lights'].append([
                int(b[0] * sx), int(b[1] * sy),
                int(b[2] * sx), int(b[3] * sy),
            ])

    # Состояние треков для отрисовки
    state = defaultdict(lambda: {
        'positions': [],
        'last_violation_t': None,
        'violation_type': '',
        'stationary_start_t': None,
        'crossed_stop': set(),
        'crossed_solid': set(),
    })

    events_dict = {}
    risk_scores = []

    frame_idx = 0
    t_sec = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_idx / fps

            # 1. Track (YOLO + ByteTrack)
            tracks = tracker.update(frame)
            detections = [
                {'bbox': t['bbox'], 'class': t['class'],
                 'confidence': t['confidence']}
                for t in tracks
            ]

            # 2. Светофор
            try:
                tl_state = tl_det.update(frame, detections)
                rules.set_traffic_light(tl_state)
            except Exception:
                tl_state = 'unknown'

            fi = {'frame_idx': frame_idx, 't_sec': t_sec}

            # 3. Rules
            try:
                rule_events = rules.process_frame(tracks, detections, fi)
            except Exception as e:
                if frame_idx % 300 == 0:
                    print(f"[rules] {type(e).__name__}: {e}")
                rule_events = []

            # 4. Events
            try:
                event_events = events_det.process_frame(tracks, detections, fi)
            except Exception as e:
                if frame_idx % 300 == 0:
                    print(f"[events] {type(e).__name__}: {e}")
                event_events = []

            # 5. Risk
            try:
                risk_score = risk_estimator.step(frame, t_sec)
            except Exception:
                risk_score = 0.0
            risk_scores.append((t_sec, float(risk_score)))

            # 6. Collect
            for ev in list(rule_events) + list(event_events):
                if len(ev) != 3:
                    continue
                s, e, et = ev
                events_dict.setdefault(et, []).append([float(s), float(e)])

            # --- Отрисовка ---
            overlay = frame.copy()
            for poly in zones['crosswalk_zones']:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
            for A, B in zones['stop_lines']:
                cv2.line(overlay, A, B, (0, 0, 255), 4)
            for A, B in zones['solid_lines']:
                cv2.line(overlay, A, B, (255, 0, 255), 4)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # Обновление состояния для отрисовки
            for tr in tracks:
                tid = tr['track_id']
                bbox = tr['bbox']
                cls = tr.get('class', '').lower()
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                st = state[tid]
                st['positions'].append((cx, cy))
                if len(st['positions']) > 120:
                    st['positions'] = st['positions'][-120:]

            for tr in tracks:
                x1, y1, x2, y2 = tr['bbox']
                cls = tr.get('class', '').lower()
                tid = tr['track_id']
                st = state[tid]
                is_viol = (st['last_violation_t'] is not None and
                           (t_sec - st['last_violation_t']) < 5.0)

                if cls in VEHICLE_CLASSES:
                    color = C_VIOLATION if is_viol else C_OK
                    label = f"#{tid} {cls}"
                    thick = 3 if is_viol else 2
                elif cls in PED_CLASSES:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    in_cross = any(bbox_intersects_polygon([x1, y1, x2, y2], poly)
                                   for poly in zones['crosswalk_zones'])
                    color = C_PED_IN if in_cross else C_PED_OUT
                    label = f"#{tid} {cls}" + ("" if in_cross else " JW")
                    thick = 2
                else:
                    color = C_UNKNOWN
                    label = f"#{tid} {cls}"
                    thick = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Светофор
            tl_colors = {'red': (0, 0, 255), 'yellow': (0, 215, 255),
                         'green': (0, 220, 0)}
            for bbox in zones['traffic_lights']:
                x1, y1, x2, y2 = bbox
                c = tl_colors.get(tl_state, (180, 180, 180))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 4)

            cv2.putText(frame, f"t={t_sec:5.1f}s  frame={frame_idx}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        # Финализация
        try:
            for ev in rules.finalize(t_sec):
                if len(ev) == 3:
                    events_dict.setdefault(ev[2], []).append(
                        [float(ev[0]), float(ev[1])])
        except Exception:
            pass
        try:
            for ev in events_det.finalize(t_sec):
                if len(ev) == 3:
                    events_dict.setdefault(ev[2], []).append(
                        [float(ev[0]), float(ev[1])])
        except Exception:
            pass
    finally:
        cap.release()
        writer.release()

    # Склейка
    final_events = []
    for et, intervals in events_dict.items():
        intervals.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for iv in intervals:
            if not merged:
                merged.append(list(iv))
            else:
                last = merged[-1]
                if iv[0] <= last[1]:
                    last[1] = max(last[1], iv[1])
                else:
                    merged.append(list(iv))
        for s, e in merged:
            final_events.append([s, e, et])
    final_events.sort(key=lambda x: (x[0], x[2]))

    return final_events, risk_scores