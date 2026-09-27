import os
import json
import numpy as np
from collections import defaultdict


# =====================================================================
# Геометрия
# =====================================================================

def _segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
            ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _point_in_polygon(pt, polygon):
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


def _bbox_intersects_polygon(bbox, poly):
    x1, y1, x2, y2 = bbox
    for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        if _point_in_polygon((cx, cy), poly):
            return True
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if _point_in_polygon((cx, cy), poly):
        return True
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
            if _segments_intersect(p1, p2, p3, p4):
                return True
    for px, py in poly_pts:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


# =====================================================================
# RulesEngine
# =====================================================================

class RulesEngine:
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'}
    PEDESTRIAN_CLASSES = {'person', 'pedestrian'}

    def __init__(
        self,
        frame_rate=30,
        zones_path='zones.json',
        risk_estimator=None,
        illegal_turn_threshold_deg=60.0,
        illegal_u_turn_threshold_deg=150.0,
        stopped_vehicle_dwell=3.0,
        congestion_vehicle_count=8,
        congestion_speed_thresh=3.0,
        failure_to_yield_distance=80.0,
        failure_to_yield_min_speed=2.0,
        event_ttl_sec=2.0,
        traffic_light_state='green',
    ):
        self.frame_rate = frame_rate
        self.risk_estimator = risk_estimator
        self.illegal_turn_threshold = np.deg2rad(illegal_turn_threshold_deg)
        self.illegal_u_turn_threshold = np.deg2rad(illegal_u_turn_threshold_deg)
        self.stopped_vehicle_dwell = stopped_vehicle_dwell
        self.congestion_vehicle_count = congestion_vehicle_count
        self.congestion_speed_thresh = congestion_speed_thresh
        self.failure_to_yield_distance = failure_to_yield_distance
        self.failure_to_yield_min_speed = failure_to_yield_min_speed
        self.event_ttl_sec = event_ttl_sec
        self.traffic_light_state = traffic_light_state

        self.stop_lines = []
        self.crosswalk_zones = []
        self.solid_lines = []
        self.no_stop_zones = []
        self._load_zones(zones_path)

        self.frame_count = 0
        self.frame_height = 0
        self.frame_width = 0

        self.track_state = defaultdict(lambda: {
            'positions': [],
            'classes': [],
            'last_seen_t': None,
            'stationary_start_t': None,
            'crossed_stop_lines': set(),
            'crossed_solid_lines': set(),
        })

        self.active_events = {}

    # ------------------------------------------------------------------

    def _load_zones(self, path):
        if not path or not os.path.exists(path):
            print(f"[RulesEngine] zones не найдены ({path})")
            return
        with open(path) as f:
            z = json.load(f)
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
        frame_idx = frame_info.get('frame_idx', self.frame_count)
        t_sec = frame_info.get('t_sec', frame_idx / self.frame_rate)

        if self.frame_height == 0:
            for d in detections:
                self.frame_width = max(self.frame_width, d['bbox'][2])
                self.frame_height = max(self.frame_height, d['bbox'][3])

        newly_detected = []
        self._update_tracks(tracks, t_sec)

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

        self._expire_events(t_sec)
        self.frame_count += 1
        return newly_detected

    def finalize(self, t_end):
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
            if len(st['positions']) > 120:
                st['positions'] = st['positions'][-120:]
                st['classes'] = st['classes'][-120:]
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
        if key not in self.active_events:
            self.active_events[key] = {
                'start_t': t_sec,
                'end_t': t_sec,
                'type': event_type,
                'last_seen_t': t_sec,
            }
            self._notify_risk(event_type)
        else:
            self.active_events[key]['end_t'] = t_sec
            self.active_events[key]['last_seen_t'] = t_sec
        ev = self.active_events[key]
        return [ev['start_t'], ev['end_t'], event_type]

    def _notify_risk(self, event_type):
        if self.risk_estimator is None:
            return
        if event_type == 'accident':
            self.risk_estimator.notify_collision()
        elif event_type == 'near_miss':
            self.risk_estimator.notify_near_miss()
        elif event_type in ('red_light', 'wrong_way', 'illegal_turn',
                             'illegal_u_turn', 'stop_line_violation',
                             'crossing_solid_lines'):
            self.risk_estimator.notify_violation(weight=0.3)
        elif event_type == 'jaywalking':
            self.risk_estimator.notify_violation(weight=0.15)
        elif event_type == 'failure_to_yield':
            self.risk_estimator.notify_violation(weight=0.25)

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
                    if self.traffic_light_state == 'red':
                        events.append(self._register(
                            'red_light', ('red_light', tid, idx), t_sec,
                        ))
                    elif self.traffic_light_state == 'yellow':
                        events.append(self._register(
                            'stop_line_violation', ('sl_yellow', tid, idx), t_sec,
                        ))
        return events

    def _check_crossing_solid_lines(self, tracks, t_sec):
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
                        'crossing_solid_lines', ('solid', tid, idx), t_sec,
                    ))
        return events

    def _check_jaywalking(self, tracks, t_sec):
        events = []
        if not self.crosswalk_zones:
            return events
        top_y = min(min(p[1] for p in poly) for poly in self.crosswalk_zones)
        bot_y = max(max(p[1] for p in poly) for poly in self.crosswalk_zones)
        left_x = min(min(p[0] for p in poly) for poly in self.crosswalk_zones)
        right_x = max(max(p[0] for p in poly) for poly in self.crosswalk_zones)

        for tr in tracks:
            if tr.get('class', '').lower() not in self.PEDESTRIAN_CLASSES:
                continue
            tid = tr['track_id']
            bbox = tr['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if cx < left_x or cx > right_x:
                continue
            if cy < top_y + 30 or cy > bot_y:
                continue
            in_any = any(
                _bbox_intersects_polygon(bbox, poly)
                for poly in self.crosswalk_zones
            )
            if not in_any:
                events.append(self._register(
                    'jaywalking', ('jw', tid), t_sec,
                ))
        return events

    def _check_failure_to_yield(self, tracks, t_sec):
        if not self.crosswalk_zones:
            return []
        events = []
        peds = []
        for tr in tracks:
            if tr.get('class', '').lower() not in self.PEDESTRIAN_CLASSES:
                continue
            st = self.track_state[tr['track_id']]
            if not st['positions']:
                continue
            pt = st['positions'][-1]
            if any(_point_in_polygon(pt, poly) for poly in self.crosswalk_zones):
                peds.append((tr['track_id'], pt))
        if not peds:
            return events

        for tr in tracks:
            if tr.get('class', '').lower() not in self.VEHICLE_CLASSES:
                continue
            tid = tr['track_id']
            st = self.track_state[tid]
            if len(st['positions']) < 2:
                continue
            v_px = float(np.hypot(
                st['positions'][-1][0] - st['positions'][-2][0],
                st['positions'][-1][1] - st['positions'][-2][1],
            ))
            if v_px < self.failure_to_yield_min_speed:
                continue
            bbox = tr['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            for ped_tid, (px, py) in peds:
                d = float(np.hypot(cx - px, cy - py))
                if d < self.failure_to_yield_distance:
                    events.append(self._register(
                        'failure_to_yield', ('fty', tid, ped_tid), t_sec,
                    ))
                    break
        return events

    def _check_wrong_way(self, tracks, t_sec):
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
            if t_sec - st['stationary_start_t'] < self.stopped_vehicle_dwell:
                continue
            bbox = tr['bbox']
            cy = (bbox[1] + bbox[3]) / 2.0
            near_stop_line = False
            for A, B in self.stop_lines:
                stop_y = (A[1] + B[1]) / 2.0
                if abs(cy - stop_y) < 60:
                    near_stop_line = True
                    break
            if near_stop_line:
                continue
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
                    'obstacle_on_road', ('obs', tuple(det['bbox'])), t_sec,
                ))
        return events

    def _check_fire_smoke(self, detections, t_sec):
        events = []
        for det in detections:
            if det.get('class', '').lower() in {'fire', 'smoke'}:
                if det.get('confidence', 0) > 0.4:
                    events.append(self._register('fire_smoke', ('fs',), t_sec))
        return events