# 🚦 Traffic Video Analysis System

Automatic traffic event detection and accident-risk estimation for fixed CCTV cameras. 

The system processes a road video and returns a list of detected events with accurate start/end timestamps, plus a per-frame accident-risk score (bonus task).

---

## 📋 Features

### Detected events (14 classes)

| # | Event | Description |
|---|---|---|
| 1 | `accident` | Collision between two vehicles |
| 2 | `near_miss` | Dangerous close approach |
| 3 | `red_light` | Running a red light |
| 4 | `wrong_way` | Driving against traffic direction |
| 5 | `illegal_u_turn` | Prohibited U-turn |
| 6 | `stopped_vehicle` | Vehicle stopped for too long |
| 7 | `jaywalking` | Pedestrian crossing outside crosswalk |
| 8 | `failure_to_yield` | Vehicle did not yield to pedestrian |
| 9 | `illegal_turn` | Prohibited turn |
| 10 | `crossing_solid_lines` | Crossing a solid lane line |
| 11 | `stop_line_violation` | Crossing a stop line on red |
| 12 | `congestion` | Traffic jam |
| 13 | `obstacle_on_road` | Static obstacle on the road |
| 14 | `fire_smoke` | Fire or smoke detected |

### Bonus — Continuous risk estimation

For every frame the system returns a risk score between `0.0` (very low) and `1.0` (very high), enabling accident anticipation up to ~5 seconds before impact.

---

## 🏗 Architecture
Video
│
▼
YOLOv8 detection
│
▼
ByteTrack tracking ──► Trajectories
│ │
│ ▼
│ ┌───────────┴────────────┐
│ ▼ ▼
│ Rules Engine Events Detector
│ (red_light, jaywalking, (accident, near_miss,
│ stopped_vehicle, ...) congestion, fire)
│ │ │
│ └───────────┬────────────┘
│ ▼
│ Events list
│ [[start, end, type], ...]
│
▼
Risk Estimator ──► float per frame (0.0–1.0)

text

---

## 📁 Project Structure
solution/
├── run_submission.py # CLI entry point
├── evaluate.py # temporal IoU evaluation
├── annotate_zones.py # interactive zone annotation
├── visualize.py # annotated video output
├── requirements.txt
├── README.md
├── .gitignore
├── zones.json # zones for traffic_short.mp4
├── zones_02.json # zones for traffic_short02.mp4
├── predictions_samples.json # example predictions
├── weights/
│ └── model.pt # YOLOv8n pretrained weights
├── src/
│ ├── init.py
│ ├── solution.py # detect_events() — main API
│ ├── detector.py # YOLO detection wrapper
│ ├── tracker.py # ByteTrack wrapper
│ ├── trajectories.py # trajectory storage
│ ├── events.py # complex events (accident, near_miss)
│ ├── rules.py # rule-based events
│ ├── risk.py # RiskEstimator class
│ └── traffic_light.py # HSV-based traffic light detector
└── notebooks/
└── EDA.ipynb # exploratory data analysis

text

---

## 🚀 Installation

**Requirements:** Python 3.10 – 3.12 (Python 3.13+ is currently not supported by PyTorch).

```bash
# 1. Clone repository
git clone <repo-url>
cd solution

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download YOLO weights (skip if weights/model.pt exists)
mkdir -p weights
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
     -o weights/model.pt
📖 Usage

Basic — detect events in Python

python
from src.solution import detect_events

events = detect_events('video.mp4')
for start, end, event_type in events:
    print(f"[{start:.2f}s → {end:.2f}s]  {event_type}")
Returns:

python
[
    [12.4, 18.9, "accident"],
    [40.0, 43.5, "red_light"],
    [72.1, 80.3, "jaywalking"],
]
CLI — run and save predictions

bash
python run_submission.py video.mp4 predictions.json
Output predictions.json:

json
{
  "video_path": "video.mp4",
  "events": [
    {"start_time": 12.4, "end_time": 18.9, "event_type": "accident"},
    {"start_time": 40.0, "end_time": 43.5, "event_type": "red_light"}
  ],
  "risk_scores": [
    {"time": 10.0, "risk": 0.02},
    {"time": 10.5, "risk": 0.03},
    {"time": 11.0, "risk": 0.08}
  ]
}
Zone annotation (one-time per camera)

Since the camera is fixed, road zones (stop lines, crosswalks, traffic lights)
are annotated once and reused:

bash
python annotate_zones.py video.mp4 zones.json
Controls:

Key	Action
1	Stop line (2 clicks)
2	Crosswalk (N clicks + Enter)
3	Solid line (2 clicks)
4	No-stop zone (N clicks + Enter)
5	Traffic light (4 clicks → bbox)
S	Save to zones.json
Q	Quit without saving
R	Undo last figure
Left-click	Add point
Right-click	Remove last point
Annotated video output

bash
python visualize.py video.mp4 zones.json
open video_annotated.mp4
Colors:

🟢 Green — vehicle driving normally
🔴 Red — vehicle committed a violation
🟣 Purple — vehicle stopped for too long
🔵 Blue — pedestrian inside crosswalk
🟠 Orange — pedestrian outside crosswalk (jaywalking)
🚦 Traffic light bbox — colored by its current state (red/yellow/green)
🎯 Evaluation

The system is evaluated with temporal IoU per event type:

bash
python evaluate.py predictions.json ground_truth.json metrics.json
Ground truth format:

json
{
  "events": [
    {"start_time": 12.4, "end_time": 18.9, "event_type": "accident"}
  ]
}
Metric:

text
IoU = sum(overlap(predicted, ground_truth)) / sum(ground_truth_duration)
🛠 Technologies

Detection: YOLOv8n (COCO, 80 classes)
Tracking: ByteTrack (built into Ultralytics)
Video I/O: OpenCV
Traffic light state: HSV color analysis
Event logic: rule-based (no additional training required)
📊 Example Results

Run on traffic_short.mp4 (60 s, 1280×720, 30 fps):

Event	Count
congestion	3
accident	5
near_miss	3
crossing_solid_lines	1
stop_line_violation	1
Full output: see predictions_samples.json.

⚙️ Configuration

Zones are defined in zones.json (and zones_02.json for the second video):

json
{
  "stop_lines":      [ [[x1,y1],[x2,y2]], ... ],
  "crosswalk_zones": [ [[x1,y1],[x2,y2],...], ... ],
  "solid_lines":     [ [[x1,y1],[x2,y2]], ... ],
  "no_stop_zones":   [ ... ],
  "traffic_lights":  [ [x1,y1,x2,y2], ... ]
}
Different videos with different camera setups use separate zones_*.json files.

📝 License

MIT

👥 Team

Abdukhalilov Abdulkhamid
Asqarov Mumin
Shokirjonova Gulzora
