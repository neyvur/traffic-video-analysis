import numpy as np
from collections import defaultdict


class EventsDetector:
    """Detects complex traffic events that require reasoning over time:
       accident / near-miss, fire/smoke (if model supports), congestion,
       obstacle on road.

    Interface (matches solution.py):
        ed = EventsDetector(frame_rate=...)
        events = ed.process_frame(tracks, detections, frame_info)
    where frame_info = {'frame_idx': int, 't_sec': float}
    Returns list of [start_t, end_t, event_type].

    Note: congestion, obstacle_on_road and fire_smoke are ALSO detected
    in RulesEngine. To avoid duplicates, this class focuses on:
      - accident / near-miss (unique to here)
      - fire / smoke (only if YOLO model has such classes)
    Congestion / obstacle checks here are minimal and can be disabled.
    """

    VEHICLE_CLASSES = {'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PEDESTRIAN_CLASSES = {'person', 'pedestrian'}
    FIRE_CLASSES = {'fire', 'smoke'}

    def __init__(
        self,
        frame_rate=30,
        # Accident thresholds
        accident_iou_thresh=0.10,
        accident_rel_vel_thresh=20.0,      # pixels / sec
        accident_slowdown_thresh=0.5,      # relative speed drop
        # Near-miss thresholds
        near_miss_distance_thresh=80.0,    # pixels between centers
        near_miss_rel_vel_thresh=15.0,
        # Congestion thresholds (light check, rules.py has heavier one)
        congestion_vehicle_count=12,
        congestion_avg_speed_thresh=2.0,   # pixels / frame
        # Event-closing: how many seconds without re-detection closes event
        event_ttl_sec=2.0,
        # Fire / smoke
        fire_smoke_conf_thresh=0.4,
    ):
        self.frame_rate = frame_rate
        self.accident_iou_thresh = accident_iou_thresh
        self.accident_rel_vel_thresh = accident_rel_vel_thresh
        self.accident_slowdown_thresh = accident_slowdown_thresh
        self.near_miss_distance_thresh = near_miss_distance_thresh
        self.near_miss_rel_vel_thresh = near_miss_rel_vel_thresh
        self.congestion_vehicle_count = congestion_vehicle_count
        self.congestion_avg_speed_thresh = congestion_avg_speed_thresh
        self.event_ttl_sec = event_ttl_sec
        self.fire_smoke_conf_thresh = fire_smoke_conf_thresh

        # Per-track history
        # tid -> {'positions': [[x,y], ...], 'bboxes': [[x1,y1,x2,y2], ...],
        #         'classes': [str, ...]}
        self.track_history = defaultdict(lambda: {
            'positions': [], 'bboxes': [], 'classes': [],
        })

        # Currently open events
        # key -> {'start_t': float, 'end_t': float, 'type': str, 'last_seen_t': float}
        self.active_events = {}

        # Already finalized events (not returned again)
        self.closed_events = []

        # Frame size, inferred from detections
        self.frame_height = 0
        self.frame_width = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, tracks, detections, frame_info):
        """Process a frame. Returns list of [start_t, end_t, event_type]."""
        frame_idx = frame_info.get('frame_idx', 0)
        t_sec = frame_info.get('t_sec', frame_idx / self.frame_rate)

        # Infer frame size
        if self.frame_height == 0:
            for d in detections:
                self.frame_width = max(self.frame_width, d['bbox'][2])
                self.frame_height = max(self.frame_height, d['bbox'][3])
            for t in tracks:
                self.frame_width = max(self.frame_width, t['bbox'][2])
                self.frame_height = max(self.frame_height, t['bbox'][3])

        # Update per-track history
        self._update_history(tracks)

        newly_detected = []

        # --- Accident / near-miss ---
        newly_detected += self._detect_accident(tracks, t_sec)

        # --- Fire / smoke (only if model produces these classes) ---
        newly_detected += self._detect_fire_smoke(detections, t_sec)

        # --- Congestion (light check, rules.py also has one) ---
        newly_detected += self._detect_congestion(tracks, t_sec)

        # --- Close events that haven't been re-detected for event_ttl_sec ---
        self._expire_events(t_sec)

        return newly_detected

    def finalize(self, t_end):
        """Close all still-open events at end of video. Returns final list."""
        out = []
        for key, ev in list(self.active_events.items()):
            ev['end_t'] = max(ev['end_t'], t_end)
            out.append([ev['start_t'], ev['end_t'], ev['type']])
        self.active_events.clear()
        return out

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _update_history(self, tracks):
        for track in tracks:
            tid = track['track_id']
            bbox = track['bbox']
            cls = track.get('class', 'unknown').lower()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            h = self.track_history[tid]
            h['positions'].append([cx, cy])
            h['bboxes'].append(list(bbox))
            h['classes'].append(cls)

            # Bound history to last ~5 seconds
            max_len = int(self.frame_rate * 5)
            if len(h['positions']) > max_len:
                h['positions'] = h['positions'][-max_len:]
                h['bboxes'] = h['bboxes'][-max_len:]
                h['classes'] = h['classes'][-max_len:]

    # ------------------------------------------------------------------
    # Helpers
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

    def _velocity(self, positions):
        """Return last-frame speed in pixels/sec."""
        if len(positions) < 2:
            return 0.0
        p1, p2 = positions[-2], positions[-1]
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return dist * self.frame_rate

    def _speed_ratio(self, positions, window=5):
        """Ratio of recent speed to earlier speed."""
        if len(positions) < 2 * window + 1:
            return 1.0
        recent = positions[-window:]
        earlier = positions[-2 * window:-window]
        v_recent = float(np.mean([
            np.hypot(recent[i + 1][0] - recent[i][0],
                     recent[i + 1][1] - recent[i][1])
            for i in range(len(recent) - 1)
        ]))
        v_early = float(np.mean([
            np.hypot(earlier[i + 1][0] - earlier[i][0],
                     earlier[i + 1][1] - earlier[i][1])
            for i in range(len(earlier) - 1)
        ]))
        if v_early < 1e-6:
            return 1.0
        return v_recent / v_early

    def _register_event(self, event_type, key, t_sec):
        """Register a continuing event. Returns [start, end, type] for output."""
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
        """Close events that haven't been re-detected recently."""
        to_close = []
        for key, ev in self.active_events.items():
            if t_sec - ev['last_seen_t'] > self.event_ttl_sec:
                to_close.append(key)
        for key in to_close:
            ev = self.active_events.pop(key)
            ev['end_t'] = max(ev['end_t'], ev['last_seen_t'])
            self.closed_events.append([ev['start_t'], ev['end_t'], ev['type']])

    # ------------------------------------------------------------------
    # Accident / near-miss
    # ------------------------------------------------------------------

    def _detect_accident(self, tracks, t_sec):
        events = []
        vehicles = [t for t in tracks if t.get('class', '').lower() in self.VEHICLE_CLASSES]

        for i, ti in enumerate(vehicles):
            for tj in vehicles[i + 1:]:
                tid_i, tid_j = ti['track_id'], tj['track_id']
                hi = self.track_history[tid_i]
                hj = self.track_history[tid_j]
                if len(hi['positions']) < 3 or len(hj['positions']) < 3:
                    continue

                # Distance between centers
                pi = hi['positions'][-1]
                pj = hj['positions'][-1]
                dist = float(np.hypot(pi[0] - pj[0], pi[1] - pj[1]))

                # IoU of current bboxes
                iou = self._iou(ti['bbox'], tj['bbox'])

                # Relative velocity
                vi = self._velocity(hi['positions'])
                vj = self._velocity(hj['positions'])
                rel_vel = abs(vi - vj)

                # Slowdown of either vehicle
                ratio_i = self._speed_ratio(hi['positions'])
                ratio_j = self._speed_ratio(hj['positions'])
                slowdown = min(ratio_i, ratio_j)

                # --- Accident condition ---
                # IoU over threshold OR very close centers, plus sudden slowdown
                is_collision = (
                    iou >= self.accident_iou_thresh
                    or dist < 50.0
                )
                is_violent = rel_vel > self.accident_rel_vel_thresh
                is_slowing = slowdown < self.accident_slowdown_thresh

                if is_collision and (is_violent or is_slowing):
                    key = ('accident', min(tid_i, tid_j), max(tid_i, tid_j))
                    events.append(self._register_event('accident', key, t_sec))
                    continue

                # --- Near-miss condition ---
                if dist < self.near_miss_distance_thresh and rel_vel > self.near_miss_rel_vel_thresh:
                    key = ('near_miss', min(tid_i, tid_j), max(tid_i, tid_j))
                    events.append(self._register_event('near_miss', key, t_sec))

        return events

    # ------------------------------------------------------------------
    # Fire / smoke
    # ------------------------------------------------------------------

    def _detect_fire_smoke(self, detections, t_sec):
        events = []
        for det in detections:
            cls = det.get('class', '').lower()
            if cls not in self.FIRE_CLASSES:
                continue
            if det.get('confidence', 0.0) < self.fire_smoke_conf_thresh:
                continue
            key = ('fire_smoke',)
            events.append(self._register_event('fire_smoke', key, t_sec))
        return events

    # ------------------------------------------------------------------
    # Congestion (light check; rules.py has a heavier one)
    # ------------------------------------------------------------------

    def _detect_congestion(self, tracks, t_sec):
        events = []
        vehicles = [t for t in tracks if t.get('class', '').lower() in self.VEHICLE_CLASSES]
        if len(vehicles) < self.congestion_vehicle_count:
            return events

        # Compute average speed across vehicles
        speeds = []
        for v in vehicles:
            tid = v['track_id']
            pos = self.track_history[tid]['positions']
            if len(pos) >= 2:
                d = float(np.hypot(pos[-1][0] - pos[-2][0], pos[-1][1] - pos[-2][1]))
                speeds.append(d)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        if avg_speed < self.congestion_avg_speed_thresh:
            key = ('congestion',)
            events.append(self._register_event('congestion', key, t_sec))
        return events