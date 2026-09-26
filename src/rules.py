import os
import json
import numpy as np
from collections import defaultdict


# =====================================================================
# Геометрия
# =====================================================================

def _segments_intersect(p1, p2, p3, p4):
    """Пересекаются ли отрезки p1-p2 и p3-p4 (в 2D)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
            ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _point_in_polygon(pt, polygon):
    """Точка внутри полигона (ray casting). polygon — список [[x,y], ...]."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside
def bbox_intersects_polygon(bbox, poly):
    """True, если bbox пересекает полигон (любой точкой)."""
    x1, y1, x2, y2 = bbox

    # 1. Углы bbox внутри полигона?
    for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        if point_in_poly((cx, cy), poly):
            return True

    # 2. Центр bbox внутри полигона?
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if point_in_poly((cx, cy), poly):
        return True

    # 3. Стороны bbox пересекают стороны полигона?
    poly_pts = [tuple(p) for p in poly]
    n = len(poly_pts)
    bbox_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for i in range(n):
        p3 = poly_pts[i]
        p4 = poly_pts[(i + 1) % n]
        for p1, p2 in bbox_edges:
            if segments_intersect(p1, p2, p3, p4):
                return True

    # 4. Полигон внутри bbox? (редкий случай — bbox больше полигона)
    for px, py in poly_pts:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    return False


# =====================================================================
# RulesEngine
# =====================================================================

class RulesEngine:
    """Rule-based traffic event detection с использованием зон из zones.json."""

    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'}
    PEDESTRIAN_CLASSES = {'person', 'pedestrian'}

    def __init__(
        self,
        frame_rate=30,
        zones_path='zones.json',
        # Пороги
        illegal_turn_threshold_deg=60.0,
        illegal_u_turn_threshold_deg=150.0,
        stopped_vehicle_dwell=3.0,
        congestion_vehicle_count=8,
        congestion_speed_thresh=3.0,
        failure_to_yield_distance=80.0,
        event_ttl_sec=2.0,
        # Состояние светофора
        traffic_light_state='green',
    ):
        self.frame_rate = frame_rate
        self.illegal_turn_threshold = np.deg2rad(illegal_turn_threshold_deg)
        self.illegal_u_turn_threshold = np.deg2rad(illegal_u_turn_threshold_deg)
        self.stopped_vehicle_dwell = stopped_vehicle_dwell
        self.congestion_vehicle_count = congestion_vehicle_count
        self.congestion_speed_thresh = congestion_speed_thresh
        self.failure_to_yield_distance = failure_to_yield_distance
        self.event_ttl_sec = event_ttl_sec
        self.traffic_light_state = traffic_light_state

        # Загрузка зон
        self.stop_lines = []
        self.crosswalk_zones = []
        self.solid_lines = []
        self.no_stop_zones = []
        self._load_zones(zones_path)

        # Состояние
        self.frame_count = 0
        self.frame_height = 0
        self.frame_width = 0

        # track_id -> {positions, classes, last_seen_t, stationary_start_t}
        self.track_state = defaultdict(lambda: {
            'positions': [],
            'classes': [],
            'last_seen_t': None,
            'stationary_start_t': None,
            # Флаги "уже зафиксировано"
            'crossed_stop_lines': set(),   # индексы стоп-линий, которые пересёк
            'crossed_solid_lines': set(),  # индексы сплошных
        })

        # Открытые события: key -> {'start_t', 'end_t', 'type', 'last_seen_t'}
        self.active_events = {}

    # ------------------------------------------------------------------

    def _load_zones(self, path):
        if not path or not os.path.exists(path):
            print(f"[RulesEngine] zones не найдены ({path}), работаем без зон")
            return
        with open(path) as f:
            z = json.load(f)
        # Конвертируем в numpy-friendly списки
        self.stop_lines = [
            (tuple(line[0]), tuple(line[1])) for line in z.get('stop_lines', [])
        ]
        self.crosswalk_zones = [
            [tuple(p) for p in poly] for poly in z.get('crosswalk_zones', [])
        ]
        self.solid_lines = [
            (tuple(line[0]), tuple(line[1])) for line in z.get('solid_lines', [])
        ]
        self.no_stop_zones = [
            [tuple(p) for p in poly] for poly in z.get('no_stop_zones', [])
        ]
        print(f"[RulesEngine] Загружено зон: "
              f"stop_lines={len(self.stop_lines)}, "
              f"crosswalks={len(self.crosswalk_zones)}, "
              f"solid_lines={len(self.solid_lines)}, "
              f"no_stop_zones={len(self.no_stop_zones)}")

    def set_traffic_light(self, state):
        self.traffic_light_state = state

    # ------------------------------------------------------------------

    def process_frame(self, tracks, detections, frame_info):
        """Возвращает список [start_t, end_t, event_type]."""
        frame_idx = frame_info.get('frame_idx', self.frame_count)
        t_sec = frame_info.get('t_sec', frame_idx / self.frame_rate)

        # Размер кадра
        if self.frame_height == 0:
            for d in detections:
                self.frame_width = max(self.frame_width, d['bbox'][2])
                self.frame_height = max(self.frame_height, d['bbox'][3])

        newly_detected = []

        # Обновляем состояние треков
        self._update_tracks(tracks, t_sec)

        # Правила
        newly_detected += self._check_stop_lines(tracks, t_sec)
        newly_detected += self._check_crossing_solid_lines(tracks, t_sec)
        newly_detected += self._check_jaywalking(tracks, t_sec)
        newly_detected += self._check_failure_to_yield(tracks, t_sec)
        newly_detected += self._check_wrong_way(tracks, t_sec)
        newly_detected += self._check_illegal_turns(tracks, t_sec)
        newly_detected += self._check_stopped_vehicle(tracks, t_sec)
        newly_detected += self._check_congestion(tracks, t_sec)
        newly_detected += self._check_obstacle_on_road(detections, t_sec)
        newly_detected += self._check_fire_smoke(detections, t_sec)

        # Закрыть старые события
        self._expire_events(t_sec)

        self.frame_count += 1
        return newly_detected

    def finalize(self, t_end):
        """Закрыть все открытые события в конце видео."""
        out = []
        for key, ev in list(self.active_events.items()):
            ev['end_t'] = max(ev['end_t'], t_end)
            out.append([ev['start_t'], ev['end_t'], ev['type']])
        self.active_events.clear()
        return out

    # ------------------------------------------------------------------

    def _update_tracks(self, tracks, t_sec):
        for tr in tracks:
            tid = tr['track_id']
            bbox = tr['bbox']
            cls = tr.get('class', 'unknown').lower()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            st = self.track_state[tid]
            st['positions'].append((cx, cy))
            st['classes'].append(cls)
            st['last_seen_t'] = t_sec

            # Ограничим историю
            if len(st['positions']) > 120:
                st['positions'] = st['positions'][-120:]
                st['classes'] = st['classes'][-120:]

            # Определение "стоит"
            if len(st['positions']) >= max(2, int(self.frame_rate)):
                recent = st['positions'][-int(self.frame_rate):]
                xs = [p[0] for p in recent]
                ys = [p[1] for p in recent]
                motion = (max(xs) - min(xs)) + (max(ys) - min(ys))
                if motion < 4.0:
                    if st['stationary_start_t'] is None:
                        st['stationary_start_t'] = t_sec
                else:
                    st['stationary_start_t'] = None
            else:
                st['stationary_start_t'] = None

    # ------------------------------------------------------------------

    def _register(self, event_type, key, t_sec):
        """Регистрирует событие (start, end), возвращает [start, end, type]."""
        if key not in self.active_events:
            self.active_events[key] = {
                'start_t': t_sec,
                'end_t': t_sec,
                'type': event_type,
                'last_seen_t': t_sec,
            }
        else:
            self.active_events[key]['end_t'] = t_sec
            self.active_events[key]['last_seen_t'] = t_sec
        ev = self.active_events[key]
        return [ev['start_t'], ev['end_t'], event_type]

    def _expire_events(self, t_sec):
        to_close = [
            k for k, ev in self.active_events.items()
            if t_sec - ev['last_seen_t'] > self.event_ttl_sec
        ]
        for k in to_close:
            self.active_events.pop(k, None)

    # ------------------------------------------------------------------
    # Правила
    # ------------------------------------------------------------------

    def _check_stop_lines(self, tracks, t_sec):
        """Пересечение стоп-линии = red_light (если красный) + stop_line_violation."""
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            st = self.track_state[tid]
            if len(st['positions']) < 2:
                continue
            p_prev = st['positions'][-2]
            p_curr = st['positions'][-1]

            for idx, (A, B) in enumerate(self.stop_lines):
                if idx in st['crossed_stop_lines']:
                    continue
                if _segments_intersect(p_prev, p_curr, A, B):
                    st['crossed_stop_lines'].add(idx)
                    # Всегда — stop_line_violation
                    events.append(self._register(
                        'stop_line_violation',
                        ('sl_violation', tid, idx),
                        t_sec,
                    ))
                    # Если красный — ещё и red_light
                    if self.traffic_light_state == 'red':
                        events.append(self._register(
                            'red_light',
                            ('red_light', tid, idx),
                            t_sec,
                        ))
        return events

    def _check_crossing_solid_lines(self, tracks, t_sec):
        """Пересечение сплошной линии."""
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            st = self.track_state[tid]
            if len(st['positions']) < 2:
                continue
            p_prev = st['positions'][-2]
            p_curr = st['positions'][-1]
            for idx, (A, B) in enumerate(self.solid_lines):
                if idx in st['crossed_solid_lines']:
                    continue
                if _segments_intersect(p_prev, p_curr, A, B):
                    st['crossed_solid_lines'].add(idx)
                    events.append(self._register(
                        'crossing_solid_lines',
                        ('solid', tid, idx),
                        t_sec,
                    ))
        return events

    def _check_jaywalking(self, tracks, t_sec):
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.PEDESTRIAN_CLASSES:
                continue
            tid = tr['track_id']
            bbox = tr['bbox']
            # bbox пешехода пересекает любую зебру?
            in_any = any(
                _bbox_intersects_polygon(bbox, poly)
                for poly in self.crosswalk_zones
            )
            # На дороге (нижняя часть кадра)?
            cy = (bbox[1] + bbox[3]) / 2.0
            on_road = self.frame_height and cy > self.frame_height * 0.35
            if on_road and not in_any:
                events.append(self._register(
                    'jaywalking', ('jw', tid), t_sec,
                ))
        return events

    def _check_failure_to_yield(self, tracks, t_sec):
        """Машина близко к пешеходу, который находится в зебре."""
        if not self.crosswalk_zones:
            return []
        events = []
        peds = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.PEDESTRIAN_CLASSES:
                continue
            tid = tr['track_id']
            st = self.track_state[tid]
            if not st['positions']:
                continue
            pt = st['positions'][-1]
            if any(_point_in_polygon(pt, poly) for poly in self.crosswalk_zones):
                peds.append((tid, pt))

        if not peds:
            return events

        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            bbox = tr['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            for ped_tid, (px, py) in peds:
                d = float(np.hypot(cx - px, cy - py))
                if d < self.failure_to_yield_distance:
                    events.append(self._register(
                        'failure_to_yield',
                        ('fty', tid, ped_tid),
                        t_sec,
                    ))
        return events

    def _check_wrong_way(self, tracks, t_sec):
        """Wrong-way: разворот траектории почти на 180° за короткое окно."""
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            pos = self.track_state[tid]['positions']
            if len(pos) < 8:
                continue
            p_early = pos[-8:-4]
            p_late = pos[-4:]
            v_early = (p_early[-1][0] - p_early[0][0],
                       p_early[-1][1] - p_early[0][1])
            v_late = (p_late[-1][0] - p_late[0][0],
                      p_late[-1][1] - p_late[0][1])
            n1 = np.hypot(*v_early)
            n2 = np.hypot(*v_late)
            if n1 < 5 or n2 < 5:
                continue
            cos = (v_early[0] * v_late[0] + v_early[1] * v_late[1]) / (n1 * n2)
            cos = max(-1.0, min(1.0, cos))
            angle = np.arccos(cos)
            if angle > np.deg2rad(120):
                events.append(self._register('wrong_way', ('ww', tid), t_sec))
        return events

    def _check_illegal_turns(self, tracks, t_sec):
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            pos = self.track_state[tid]['positions']
            if len(pos) < 6:
                continue
            p1, p2, p3 = pos[-3], pos[-2], pos[-1]
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            n1 = np.hypot(*v1)
            n2 = np.hypot(*v2)
            if n1 < 3 or n2 < 3:
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            cos = max(-1.0, min(1.0, cos))
            angle = np.arccos(cos)
            if angle > self.illegal_u_turn_threshold:
                events.append(self._register(
                    'illegal_u_turn', ('ut', tid), t_sec))
            elif angle > self.illegal_turn_threshold:
                events.append(self._register(
                    'illegal_turn', ('it', tid), t_sec))
        return events

    def _check_stopped_vehicle(self, tracks, t_sec):
        events = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            st = self.track_state[tid]
            if st['stationary_start_t'] is None:
                continue
            if t_sec - st['stationary_start_t'] >= self.stopped_vehicle_dwell:
                events.append(self._register(
                    'stopped_vehicle', ('stop', tid), t_sec))
        return events

    def _check_congestion(self, tracks, t_sec):
        events = []
        vehicles = [
            t for t in tracks
            if t.get('class', '').lower() in self.VEHICLE_CLASSES
        ]
        if len(vehicles) < self.congestion_vehicle_count:
            return events
        speeds = []
        for v in vehicles:
            pos = self.track_state[v['track_id']]['positions']
            if len(pos) >= 2:
                d = float(np.hypot(pos[-1][0] - pos[-2][0],
                                   pos[-1][1] - pos[-2][1]))
                speeds.append(d)
        avg = float(np.mean(speeds)) if speeds else 0.0
        if avg < self.congestion_speed_thresh:
            events.append(self._register('congestion', ('cong',), t_sec))
        return events

    def _check_obstacle_on_road(self, detections, t_sec):
        events = []
        if not self.frame_height:
            return events
        for det in detections:
            cls = det.get('class', '').lower()
            if cls in self.VEHICLE_CLASSES or cls in self.PEDESTRIAN_CLASSES:
                continue
            if cls in {'fire', 'smoke', 'traffic light', 'traffic_light', 'stop sign'}:
                continue
            cy = (det['bbox'][1] + det['bbox'][3]) / 2.0
            if cy > self.frame_height * 0.5:
                events.append(self._register(
                    'obstacle_on_road',
                    ('obs', tuple(det['bbox'])),
                    t_sec,
                ))
        return events

    def _check_fire_smoke(self, detections, t_sec):
        events = []
        for det in detections:
            if det.get('class', '').lower() in {'fire', 'smoke'}:
                if det.get('confidence', 0) > 0.4:
                    events.append(self._register('fire_smoke', ('fs',), t_sec))
        return events