import cv2
import numpy as np
import json
from ultralytics import YOLO

from .tracker import ByteTrack
from .trajectories import Trajectories
from .events import EventsDetector
from .rules import RulesEngine
from .risk import RiskEstimator
from .traffic_light import TrafficLightDetector


def detect_events(video_path: str, zones_path: str = 'zones.json',
                  return_risk: bool = False):
    """Detect traffic events in a video and return list of
    [start_time, end_time, event_type].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ([], []) if return_risk else []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if not frame_rate or frame_rate <= 0:
        frame_rate = 30.0

    # --- Модули ---
        # --- Модули ---
    yolo = YOLO('weights/model.pt')
    tracker = ByteTrack(yolo=yolo, frame_rate=frame_rate, conf_thres=0.25)
    trajectories = Trajectories()
    risk_estimator = RiskEstimator(frame_rate=frame_rate)
    events_detector = EventsDetector(
        frame_rate=frame_rate,
        risk_estimator=risk_estimator,
        congestion_vehicle_count=999,
        congestion_avg_speed_thresh=0.1,
    )
    rules_engine = RulesEngine(
        frame_rate=frame_rate,
        zones_path=zones_path,
        risk_estimator=risk_estimator,
        congestion_vehicle_count=999,
        congestion_speed_thresh=0.1,
    )
    tl_detector = TrafficLightDetector(frame_rate=frame_rate,
                                        zones_path=zones_path)

    events_dict = {}
    risk_scores = []

    frame_idx = 0
    t_sec = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / frame_rate

            tracks = tracker.update(frame)
            detections = [
                {'bbox': t['bbox'], 'class': t['class'],
                 'confidence': t['confidence']}
                for t in tracks
            ]
            trajectories.update(tracks)

            try:
                tl_state = tl_detector.update(frame, detections)
                rules_engine.set_traffic_light(tl_state)
            except Exception as e:
                if frame_idx < 5 or frame_idx % 300 == 0:
                    print(f"[tl] frame {frame_idx}: {type(e).__name__}: {e}")

            frame_info = {'frame_idx': frame_idx, 't_sec': t_sec}

            try:
                rule_events = rules_engine.process_frame(tracks, detections,
                                                        frame_info)
            except Exception as e:
                if frame_idx < 5 or frame_idx % 300 == 0:
                    print(f"[rules] frame {frame_idx}: {type(e).__name__}: {e}")
                rule_events = []

            try:
                event_events = events_detector.process_frame(tracks, detections,
                                                            frame_info)
            except Exception as e:
                if frame_idx < 5 or frame_idx % 300 == 0:
                    print(f"[events] frame {frame_idx}: {type(e).__name__}: {e}")
                event_events = []

            try:
                risk_score = risk_estimator.step(frame, t_sec)
            except Exception as e:
                if frame_idx < 5 or frame_idx % 300 == 0:
                    print(f"[risk] frame {frame_idx}: {type(e).__name__}: {e}")
                risk_score = 0.0
            risk_scores.append((t_sec, float(risk_score)))

            for ev in list(rule_events) + list(event_events):
                if len(ev) != 3:
                    continue
                s, e, et = ev
                events_dict.setdefault(et, []).append([float(s), float(e)])

            frame_idx += 1

        try:
            for ev in rules_engine.finalize(t_sec):
                if len(ev) == 3:
                    events_dict.setdefault(ev[2], []).append(
                        [float(ev[0]), float(ev[1])])
        except Exception as e:
            print(f"[finalize-rules] {type(e).__name__}: {e}")
        try:
            for ev in events_detector.finalize(t_sec):
                if len(ev) == 3:
                    events_dict.setdefault(ev[2], []).append(
                        [float(ev[0]), float(ev[1])])
        except Exception as e:
            print(f"[finalize-events] {type(e).__name__}: {e}")
    finally:
        cap.release()

    # --- Склейка ---
    final_events = []
    for event_type, intervals in events_dict.items():
        intervals.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(list(interval))
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    last[1] = max(last[1], interval[1])
                else:
                    merged.append(list(interval))
        for s, e in merged:
            if e < s:
                s, e = e, s
            final_events.append([s, e, event_type])

    final_events.sort(key=lambda x: (x[0], x[2]))

    if return_risk:
        return final_events, risk_scores
    return final_events


def run_submission(video_path: str, output_path: str = None,
                   zones_path: str = 'zones.json'):
    """Запускает анализ и сохраняет predictions.json."""
    events, risk_scores = detect_events(video_path, zones_path=zones_path,
                                         return_risk=True)

    predictions = {
        'video_path': video_path,
        'events': [
            {'start_time': e[0], 'end_time': e[1], 'event_type': e[2]}
            for e in events
        ],
        'risk_scores': [{'time': t, 'risk': r} for t, r in risk_scores],
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

    return predictions


def evaluate(predictions_path: str, ground_truth_path: str):
    """Оценка предсказаний против ground truth."""
    with open(predictions_path) as f:
        predictions = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    pred_events = (predictions.get('events', [])
                   if isinstance(predictions, dict) else predictions)
    gt_events = (ground_truth.get('events', [])
                 if isinstance(ground_truth, dict) else ground_truth)

    event_types = set()
    for e in pred_events:
        event_types.add(e['event_type'])
    for e in gt_events:
        event_types.add(e['event_type'])

    metrics = {}
    for event_type in sorted(event_types):
        pred_intervals = [(e['start_time'], e['end_time'])
                          for e in pred_events
                          if e['event_type'] == event_type]
        gt_intervals = [(e['start_time'], e['end_time'])
                        for e in gt_events
                        if e['event_type'] == event_type]

        total_overlap = 0.0
        total_gt_duration = 0.0
        for gt_start, gt_end in gt_intervals:
            total_gt_duration += (gt_end - gt_start)
            for ps_start, ps_end in pred_intervals:
                overlap = max(0.0, min(gt_end, ps_end) - max(gt_start, ps_start))
                total_overlap += overlap

        iou = total_overlap / total_gt_duration if total_gt_duration > 0 else 0.0
        metrics[event_type] = {
            'iou': round(iou, 4),
            'predicted_count': len(pred_intervals),
            'ground_truth_count': len(gt_intervals),
        }

    return metrics