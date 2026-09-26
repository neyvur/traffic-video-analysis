import numpy as np
from collections import defaultdict

class Trajectories:
    """Track vehicle and pedestrian trajectories for analysis"""
    
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.trajectories = defaultdict(lambda: {'positions': [], 'classes': []})
    
    def update(self, tracks):
        """Update trajectories with current track positions"""
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            traj = self.trajectories[track_id]
            traj['positions'].append([center_x, center_y])
            traj['classes'].append(track.get('class', 'unknown'))
            
            # Keep history bounded
            if len(traj['positions']) > self.max_history:
                traj['positions'] = traj['positions'][-self.max_history:]
                traj['classes'] = traj['classes'][-self.max_history:]
    
    def get_trajectory(self, track_id):
        """Get position history for a track"""
        return self.trajectories.get(track_id, {'positions': [], 'classes': []})
    
    def get_all_positions(self):
        """Get all tracked positions with IDs and classes"""
        result = []
        for track_id, data in self.trajectories.items():
            result.append({
                'track_id': track_id,
                'positions': data['positions'],
                'classes': data['classes']
            })
        return result