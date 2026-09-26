"""
api/annotate.py — прогон пайплайна с записью аннотированного видео.

Функция process_video_annotated():
  - YOLO + ByteTrack
  - RulesEngine + EventsDetector
  - Рисует bbox, track_id, события, светофор
  - Пишет в output_path
  - Возвращает список событий [[start, end, type], ...]
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


def process_video_annotated(video_path, output_path, zones_path='zones.json',
                            conf_thres=0.25):
    """Прогон + запись аннотированного видео. Возвращает events list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter — mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Модели
    yolo = YOLO('weights/model.pt')
    tracker = ByteTrack(yolo=yolo, frame_rate=fps, conf_thres=conf_thres)
    rules = RulesEngine(frame_rate=fps, zones_path=zones_path)
    events_det = EventsDetector(frame_rate=fps)
    tl_det = TrafficLightDetector(frame_rate=fps, zones_path=zones_path)

    # Зоны
    zones = {'stop_lines': [], 'crosswalk_zones': [], 'solid_lines': [],
             'traffic_lights': []}
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            z = json.load(f)
        zones['stop_lines'] = [(tuple(l[0]), tuple(l[1]))
                               for l in z.get('stop_lines', [])]
        zones['crosswalk_zones'] = list(z.get('crosswalk_zones', []))
        zones['solid_lines'] = [(tuple(l[0]), tuple(l[1]))
                                for l in z.get('solid_lines', [])]
        for tl in z.get('traffic_lights', []):
            if isinstance(tl, dict) and 'bbox' in tl:
                b = tl['bbox']
                zones['traffic_lights'].append(
                    [int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            elif isinstance(tl, list) and len(tl) == 4:
                zones['traffic_lights'].append(
                    [int(tl[0]), int(tl[1]), int(tl[2]), int(tl[3])])

    state = defaultdict(lambda: {
        'positions': [], 'last_violation_t': None, 'violation_type': '',
        'stationary_start_t': None, 'crossed_stop': set(),
        'crossed_stop_t': None, 'crossed_solid': set(),
    })

    events_dict = {}

    frame_idx = 0
    t_sec = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_idx / fps

            tracks = tracker.update(frame)
            detections = [{'bbox': t['bbox'], 'class': t['class'],
                           'confidence': t['confidence']} for t in tracks]

            # светофор
            tl_state = tl_det.update(frame, detections)
            rules.set_traffic_light(tl_state)

            fi = {'frame_idx': frame_idx, 't_sec': t_sec}
            try:
                rule_events = rules.process_frame(tracks, detections, fi)
            except Exception:
                rule_events = []
            try:
                ev_events = events_det.process_frame(tracks, detections, fi)
            except Exception:
                ev_events = []

            for ev in list(rule_events) + list(ev_events):
                if len(ev) == 3:
                    s, e, et = ev
                    events_dict.setdefault(et, []).append([float(s), float(e)])

            # --- Локальная логика для отрисовки ---
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

                if len(st['positions']) >= int(fps):
                    recent = st['positions'][-int(fps):]
                    xs = [p[0] for p in recent]
                    ys = [p[1] for p in recent]
                    motion = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    if motion < 8.0:
                        if st['stationary_start_t'] is None:
                            st['stationary_start_t'] = t_sec
                    else:
                        st['stationary_start_t'] = None
                else:
                    st['stationary_start_t'] = None

                if cls in VEHICLE_CLASSES and len(st['positions']) >= 2:
                    p_prev = st['positions'][-2]
                    p_curr = st['positions'][-1]
                    for idx, (A, B) in enumerate(zones['stop_lines']):
                        if idx in st['crossed_stop']:
                            continue
                        if segments_intersect(p_prev, p_curr, A, B):
                            st['crossed_stop'].add(idx)
                            st['crossed_stop_t'] = t_sec
                            if tl_state == 'red':
                                st['last_violation_t'] = t_sec
                                st['violation_type'] = 'RED LIGHT'
                    for idx, (A, B) in enumerate(zones['solid_lines']):
                        if idx in st['crossed_solid']:
                            continue
                        if segments_intersect(p_prev, p_curr, A, B):
                            st['crossed_solid'].add(idx)
                            st['last_violation_t'] = t_sec
                            st['violation_type'] = 'SOLID LINE'

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

            for tr in tracks:
                x1, y1, x2, y2 = tr['bbox']
                cls = tr.get('class', '').lower()
                tid = tr['track_id']
                st = state[tid]
                is_viol = (st['last_violation_t'] is not None and
                           (t_sec - st['last_violation_t']) < 5.0)

                if cls in VEHICLE_CLASSES:
                    if is_viol:
                        color = C_VIOLATION
                        label = f"#{tid} {cls} {st['violation_type']}"
                        thick = 3
                    elif (st['stationary_start_t'] is not None and
                          (t_sec - st['stationary_start_t']) >= 2.0):
                        color = C_STOPPED
                        label = f"#{tid} {cls} STOPPED"
                        thick = 2
                    else:
                        color = C_OK
                        label = f"#{tid} {cls}"
                        thick = 2
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

            # Светофор из zones.json
            for bbox in zones['traffic_lights']:
                x1, y1, x2, y2 = bbox
                c = {'red': (0, 0, 255), 'yellow': (0, 215, 255),
                     'green': (0, 220, 0)}.get(tl_state, (180, 180, 180))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 4)

            # Плашка с временем
            cv2.putText(frame, f"t={t_sec:5.1f}s  frame={frame_idx}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        # Закрытие открытых
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
    return final_events
