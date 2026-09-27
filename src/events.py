import numpy as np
from collections import defaultdict


class EventsDetector:
    """Detects complex traffic events: accident / near-miss, fire, congestion."""

    VEHICLE_CLASSES = {'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PEDESTRIAN_CLASSES = {'person', 'pedestrian'}
    FIRE_CLASSES = {'fire', 'smoke'}

    def __init__(
        self,
        frame_rate=30,
        risk_estimator=None,
        accident_dist_thresh=60.0,
        accident_approach_thresh=5.0,
        accident_slowdown_thresh=0.4,
        accident_max_duration=10.0,
        near_miss_dist_thresh=120.0,
        near_miss_approach_thresh=8.0,
        near_miss_max_duration=3.0,
        congestion_vehicle_count=12,
        congestion_avg_speed_thresh=2.0,
        congestion_max_duration=30.0,
        event_ttl_sec=2.0,
        velocity_window=5,
        fire_smoke_conf_thresh=0.4,
    ):
        self.frame_rate = frame_rate
        self.risk_estimator = risk_estimator
        self.accident_dist_thresh = accident_dist_thresh
        self.accident_approach_thresh = accident_approach_thresh
        self.accident_slowdown_thresh = accident_slowdown_thresh
        self.accident_max_duration = accident_max_duration
        self.near_miss_dist_thresh = near_miss_dist_thresh
        self.near_miss_approach_thresh = near_miss_approach_thresh
        self.near_miss_max_duration = near_miss_max_duration
        self.congestion_vehicle_count = congestion_vehicle_count
        self.congestion_avg_speed_thresh = congestion_avg_speed_thresh
        self.congestion_max_duration = congestion_max_duration
        self.event_ttl_sec = event_ttl_sec
        self.velocity_window = velocity_window
        self.fire_smoke_conf_thresh = fire_smoke_conf_thresh

        self.track_history = defaultdict(lambda: {
            'positions': [], 'bboxes': [], 'classes': [],
        })
        self.active_events = {}

    # ------------------------------------------------------------------

    def process_frame(self, tracks, detections, frame_info):
        frame_idx = frame_info.get('frame_idx', 0)
        t_sec = frame_info.get('t_sec', frame_idx / self.frame_rate)

        self._update_history(tracks)

        newly_detected = []
        newly_detected += self._detect_accident(t_sec)
        newly_detected += self._detect_fire_smoke(detections, t_sec)
        newly_detected += self._detect_congestion(tracks, t_sec)

        self._cap_durations(t_sec)
        self._expire_events(t_sec)
        return newly_detected

    def finalize(self, t_end):
        out = []
        for key, ev in list(self.active_events.items()):
            ev['end_t'] = max(ev['end_t'], t_end)
            out.append([ev['start_t'], ev['end_t'], ev['type']])
        self.active_events.clear()
        return out

    # ------------------------------------------------------------------

    def _update_history(self, tracks):
        for tr in tracks:
            tid = tr['track_id']
            bbox = tr['bbox']
            cls = tr.get('class', 'unknown').lower()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            h = self.track_history[tid]
            h['positions'].append([cx, cy])
            h['bboxes'].append(list(bbox))
            h['classes'].append(cls)
            max_len = int(self.frame_rate * 5)
            if len(h['positions']) > max_len:
                h['positions'] = h['positions'][-max_len:]
                h['bboxes'] = h['bboxes'][-max_len:]
                h['classes'] = h['classes'][-max_len:]

    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _approach_speed(self, pi_hist, pj_hist, window=5):
        if len(pi_hist) < window + 1 or len(pj_hist) < window + 1:
            return 0.0
        p_i0 = pi_hist[-(window + 1)]
        p_j0 = pj_hist[-(window + 1)]
        d0 = float(np.hypot(p_i0[0] - p_j0[0], p_i0[1] - p_j0[1]))
        p_i1 = pi_hist[-1]
        p_j1 = pj_hist[-1]
        d1 = float(np.hypot(p_i1[0] - p_j1[0], p_i1[1] - p_j1[1]))
        return (d0 - d1) / window

    def _speed_ratio(self, positions, window=5):
        if len(positions) < 2 * window + 1:
            return 1.0
        recent = positions[-window:]
        earlier = positions[-2 * window:-window]
        v_recent = float(np.mean([
            np.hypot(recent[i + 1][0] - recent[i][0],
                     recent[i + 1][1] - recent[i][1])
            for i in range(len(recent) - 1)
        ])) if len(recent) > 1 else 0.0
        v_early = float(np.mean([
            np.hypot(earlier[i + 1][0] - earlier[i][0],
                     earlier[i + 1][1] - earlier[i][1])
            for i in range(len(earlier) - 1)
        ])) if len(earlier) > 1 else 0.0
        if v_early < 1e-3:
            return 1.0
        return v_recent / v_early

    # ------------------------------------------------------------------

    def _register_event(self, event_type, key, t_sec):
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

    def _expire_events(self, t_sec):
        to_close = [
            k for k, ev in self.active_events.items()
            if t_sec - ev['last_seen_t'] > self.event_ttl_sec
        ]
        for k in to_close:
            self.active_events.pop(k, None)

    def _cap_durations(self, t_sec):
        max_dur = {
            'accident': self.accident_max_duration,
            'near_miss': self.near_miss_max_duration,
            'congestion': self.congestion_max_duration,
        }
        to_close = []
        for key, ev in self.active_events.items():
            limit = max_dur.get(ev['type'])
            if limit is not None and (t_sec - ev['start_t']) > limit:
                if t_sec - ev['last_seen_t'] <= self.event_ttl_sec:
                    ev['start_t'] = t_sec
                else:
                    to_close.append(key)
        for k in to_close:
            self.active_events.pop(k, None)

    # ------------------------------------------------------------------
    # Accident / Near-miss
    # ------------------------------------------------------------------

    def _detect_accident(self, t_sec):
        events = []
        tids = list(self.track_history.keys())

        for i, tid_i in enumerate(tids):
            hi = self.track_history[tid_i]
            if not hi['classes'] or hi['classes'][-1] not in self.VEHICLE_CLASSES:
                continue
            for tid_j in tids[i + 1:]:
                hj = self.track_history[tid_j]
                if not hj['classes'] or hj['classes'][-1] not in self.VEHICLE_CLASSES:
                    continue
                if len(hi['positions']) < self.velocity_window + 1:
                    continue
                if len(hj['positions']) < self.velocity_window + 1:
                    continue

                pi = hi['positions'][-1]
                pj = hj['positions'][-1]
                dist = float(np.hypot(pi[0] - pj[0], pi[1] - pj[1]))
                approach = self._approach_speed(hi['positions'], hj['positions'])

                ratio_i = self._speed_ratio(hi['positions'])
                ratio_j = self._speed_ratio(hj['positions'])
                slowdown = min(ratio_i, ratio_j)

                # --- Accident ---
                if (dist < self.accident_dist_thresh
                        and approach > self.accident_approach_thresh
                        and slowdown < self.accident_slowdown_thresh):
                    key = ('accident', min(tid_i, tid_j), max(tid_i, tid_j))
                    events.append(self._register_event('accident', key, t_sec))
                    continue

                # --- Near-miss ---
                if (dist < self.near_miss_dist_thresh
                        and approach > self.near_miss_approach_thresh):
                    key = ('near_miss', min(tid_i, tid_j), max(tid_i, tid_j))
                    events.append(self._register_event('near_miss', key, t_sec))

        return events

    # ------------------------------------------------------------------

    def _detect_fire_smoke(self, detections, t_sec):
        events = []
        for det in detections:
            cls = det.get('class', '').lower()
            if cls in self.FIRE_CLASSES:
                if det.get('confidence', 0.0) >= self.fire_smoke_conf_thresh:
                    key = ('fire_smoke',)
                    events.append(self._register_event('fire_smoke', key, t_sec))
        return events

    def _detect_congestion(self, tracks, t_sec):
        events = []
        vehicles = [
            t for t in tracks
            if t.get('class', '').lower() in self.VEHICLE_CLASSES
        ]
        if len(vehicles) < self.congestion_vehicle_count:
            return events

        speeds = []
        for v in vehicles:
            pos = self.track_history[v['track_id']]['positions']
            if len(pos) >= 2:
                speeds.append(float(np.hypot(
                    pos[-1][0] - pos[-2][0],
                    pos[-1][1] - pos[-2][1],
                )))
        avg = float(np.mean(speeds)) if speeds else 0.0
        if avg < self.congestion_avg_speed_thresh:
            key = ('congestion',)
            events.append(self._register_event('congestion', key, t_sec))
        return events