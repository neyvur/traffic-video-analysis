"""
risk.py — Behavioral accident-risk estimator.

Risk is driven by:
    - rule violations (red_light, wrong_way, stop_line, ...)
    - near-miss events
    - actual collisions
    - traffic density (crowding)

The estimator receives events via notify_* hooks called from RulesEngine
and EventsDetector. No brightness-variance heuristics.
"""
import numpy as np
from collections import deque


class RiskEstimator:
    """Continuous accident risk estimation (0.0 – 1.0 per frame).

    Uses only information from frames already seen (no future peeking).
    """

    def __init__(self, frame_rate=30):
        self.frame_rate = frame_rate
        self.reset()

    def reset(self, meta=None):
        self.meta = meta or {}
        self.violation_weight = 0.0
        self.near_miss_weight = 0.0
        self.collision_weight = 0.0
        self.crowding_weight = 0.0
        self.last_decay_t = 0.0
        self.history = deque(maxlen=30)
        self.risk_score = 0.0

    # ------------------------------------------------------------------
    # Hooks — RulesEngine / EventsDetector call these when events happen
    # ------------------------------------------------------------------

    def notify_violation(self, weight=0.3):
        """Rule violation: red light, wrong-way, stop-line, etc."""
        self.violation_weight = min(1.0, self.violation_weight + weight)

    def notify_near_miss(self, weight=0.5):
        """Dangerous close approach between two objects."""
        self.near_miss_weight = min(1.0, self.near_miss_weight + weight)

    def notify_collision(self, weight=1.0):
        """Actual collision detected."""
        self.collision_weight = 1.0
        self.near_miss_weight = 1.0

    def notify_crowding(self, level):
        """Traffic density level (0.0 – 1.0), from vehicle count."""
        self.crowding_weight = max(self.crowding_weight, float(level))

    # ------------------------------------------------------------------

    def step(self, frame, t_sec):
        """Return risk score 0.0 – 1.0 for the current frame."""
        # 1. Decay accumulated weights every 0.5s
        if t_sec - self.last_decay_t >= 0.5:
            self.violation_weight *= 0.85
            self.near_miss_weight *= 0.85
            self.crowding_weight *= 0.95
            self.collision_weight *= 0.95
            self.last_decay_t = t_sec

        # 2. Combine sources
        combined = (
            0.5 * self.collision_weight +
            0.3 * self.near_miss_weight +
            0.15 * self.violation_weight +
            0.05 * self.crowding_weight
        )

        # 3. Smooth over ~1s
        self.history.append(combined)
        smoothed = float(np.mean(self.history)) if self.history else combined
        self.risk_score = float(np.clip(smoothed, 0.0, 1.0))
        return self.risk_score

    def get_state(self):
        return {
            'risk': self.risk_score,
            'collision': self.collision_weight,
            'near_miss': self.near_miss_weight,
            'violation': self.violation_weight,
            'crowding': self.crowding_weight,
        }