import cv2
import numpy as np
from collections import defaultdict


class RiskEstimator:
    """Continuous accident risk estimation.
    
    Returns a risk score between 0.0 (very low) and 1.0 (very high) for each frame.
    Uses only information from frames already seen (no future peeking).
    """
    
    def __init__(self, frame_rate=30):
        self.frame_rate = frame_rate
        self.reset()
    
    def reset(self, meta=None):
        """Reset estimator with metadata about the scene."""
        self.risk_score = 0.0
        self.event_weight = 0.0        # accumulated event influence
        self.behavioral_risk = 0.0     # from rule violations
        self.trajectory_risk = 0.0     # from erratic movement
        self.last_reset = None
        self.meta = meta or {}
        # Per-track behavioral scores
        self.track_risk = defaultdict(float)
        # History for smoothing
        self.risk_history = []
        self.max_history = 30          # 1 second at 30 fps
    
    def step(self, frame, t_sec):
        """Process a frame and return risk score 0.0 to 1.0.
        
        Args:
            frame: numpy array (BGR) or None if just updating state
            t_sec: current time in seconds
            
        Returns:
            float in [0.0, 1.0] — current accident risk estimate
        """
        # --- Base risk from frame characteristics ---
        if frame is not None:
            try:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                # Variance of brightness — proxy for visual "chaos"
                variance = float(np.var(gray))
                chaos_risk = min(1.0, variance / 5000.0)
                
                # Crowding metric — number of tracked objects with nonzero risk
                active_tracks = sum(1 for v in self.track_risk.values() if v > 0.1)
                crowding_metric = min(1.0, active_tracks / 20.0)
                
                # Weighted combination
                frame_risk = 0.6 * chaos_risk + 0.4 * crowding_metric
            except Exception:
                frame_risk = 0.0
        else:
            frame_risk = 0.0
        
        # --- Behavioral risk from rules ---
        behavioral = min(1.0, self.behavioral_risk)
        
        # --- Trajectory risk (erratic movement) ---
        trajectory = min(1.0, self.trajectory_risk)
        
        # --- Combine all sources ---
        # Frame risk is a baseline; behavioral and trajectory can push it up
        combined = (
            0.4 * frame_risk +
            0.4 * behavioral +
            0.2 * trajectory
        )
        
        # Smooth over recent history to avoid jitter
        self.risk_history.append(combined)
        if len(self.risk_history) > self.max_history:
            self.risk_history = self.risk_history[-self.max_history:]
        
        smoothed = float(np.mean(self.risk_history)) if self.risk_history else combined
        
        # Blend instantaneous and smoothed
        self.risk_score = 0.5 * combined + 0.5 * smoothed
        
        # Decay behavioral/trajectory risk over time (if no new events)
        self.behavioral_risk *= 0.98
        self.trajectory_risk *= 0.98
        
        # Clamp
        self.risk_score = max(0.0, min(1.0, self.risk_score))
        return float(self.risk_score)
    
    # --- Optional hooks that rules/events can call to raise risk ---
    
    def notify_violation(self, weight=0.3):
        """Called when a rule violation is detected (red light, wrong way, etc.)."""
        self.behavioral_risk = min(1.0, self.behavioral_risk + weight)
    
    def notify_near_miss(self, weight=0.5):
        """Called when two objects have a near-miss (close approach)."""
        self.trajectory_risk = min(1.0, self.trajectory_risk + weight)
    
    def notify_collision(self, weight=1.0):
        """Called when a collision is actually detected."""
        self.behavioral_risk = 1.0
        self.trajectory_risk = 1.0