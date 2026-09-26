import sys
import os
import json
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from src.tracker import ByteTrack
from src.traffic_light import TrafficLightDetector

def _segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
            ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _point_in_polygon(pt, poly):
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


# ---------- Цвета ----------
C_OK        = (0, 220, 0)
C_VIOLATION = (0, 0, 255)
C_STOPPED   = (200, 0, 200)
C_PED_IN    = (255, 180, 0)
C_PED_OUT   = (0, 165, 255)
C_UNKNOWN   = (180, 180, 180)
C_TL_RED    = (0, 0, 255)
C_TL_YEL    = (0, 215, 255)
C_TL_GRN    = (0, 220, 0)
C_TL_UNK    = (180, 180, 180)

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
PED_CLASSES = {'person', 'pedestrian'}


# ---------- Геометрия ----------
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


def tl_color(s):
    return {'red': C_TL_RED, 'yellow': C_TL_YEL,
            'green': C_TL_GRN}.get(s, C_TL_UNK)


# =====================================================================
# Основной проход
# =====================================================================

def process_video(video_path, output_path, zones_path='zones.json',
                  conf_thres=0.25,
                  violation_ttl=5.0, stopped_dwell=2.0,
                  tl_stable_frames=15, tl_wait_after_change=10):
    """
    Args:
        violation_ttl: сколько секунд машина остаётся красной после нарушения
        stopped_dwell: сколько секунд стоять, чтобы считаться "STOPPED"
        tl_stable_frames: сколько кадров цвет светофора должен быть стабилен
        tl_wait_after_change: сколько секунд после смены светофора НЕ фиксировать нарушения
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не открыть {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"=== {video_path} ===")
    print(f"FPS={fps:.2f}  {w}x{h}  frames={n_frames}  dur={n_frames/fps:.1f}s")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))

    # YOLO + ByteTrack
    yolo = YOLO('weights/model.pt')
    tracker = ByteTrack(yolo=yolo, frame_rate=fps, conf_thres=conf_thres)
    tl_detector = TrafficLightDetector(frame_rate=fps, zones_path=zones_path)

    # ---- Зоны ----
    zones = {'stop_lines': [], 'crosswalk_zones': [],
             'solid_lines': [], 'no_stop_zones': [],
             'traffic_lights': []}
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            z = json.load(f)
        zones['stop_lines'] = [(tuple(l[0]), tuple(l[1]))
                               for l in z.get('stop_lines', [])]
        zones['crosswalk_zones'] = list(z.get('crosswalk_zones', []))
        zones['solid_lines'] = [(tuple(l[0]), tuple(l[1]))
                                for l in z.get('solid_lines', [])]
        zones['no_stop_zones'] = list(z.get('no_stop_zones', []))
        # traffic_lights
        for tl in z.get('traffic_lights', []):
            if isinstance(tl, dict) and 'bbox' in tl:
                b = tl['bbox']
                zones['traffic_lights'].append(
                    [int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            elif isinstance(tl, list) and len(tl) == 4:
                zones['traffic_lights'].append(
                    [int(tl[0]), int(tl[1]), int(tl[2]), int(tl[3])])

    print(f"Зоны: stop={len(zones['stop_lines'])}, "
          f"crosswalk={len(zones['crosswalk_zones'])}, "
          f"solid={len(zones['solid_lines'])}, "
          f"TL={len(zones['traffic_lights'])}")

    # ---- Состояние треков ----
    state = defaultdict(lambda: {
        'positions': [],
        'last_violation_t': None,
        'violation_type': '',
        'stationary_start_t': None,
        'crossed_stop': set(),
        'crossed_stop_t': None,
        'crossed_solid': set(),
        'class': '',
    })

    # ---- Стабилизация светофора ----
    tl_stable_history = []          # последние tl_stable_frames состояний
    tl_stable = 'unknown'            # стабильное состояние
    tl_last_change_t = None          # когда светофор последний раз сменился

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        # --- YOLO + ByteTrack ---
        tracks = tracker.update(frame)

        # --- Светофор ---
        tl_dets = [{'bbox': t['bbox'], 'class': t['class']}
                   for t in tracks]
        tl_state = tl_detector.update(frame, tl_dets)

        # Обновляем историю стабильности
        tl_stable_history.append(tl_state)
        if len(tl_stable_history) > tl_stable_frames:
            tl_stable_history = tl_stable_history[-tl_stable_frames:]

        # Определяем стабильное состояние
        prev_stable = tl_stable
        if len(tl_stable_history) == tl_stable_frames:
            if all(s == tl_stable_history[0] for s in tl_stable_history):
                tl_stable = tl_stable_history[0]
            else:
                tl_stable = 'unknown'

        # Фиксируем момент смены устойчивого состояния
        if tl_stable != prev_stable and tl_stable in ('red', 'yellow', 'green'):
            tl_last_change_t = t_sec

        # --- Обновление состояния треков ---
        for tr in tracks:
            tid = tr['track_id']
            bbox = tr['bbox']
            cls = tr.get('class', '').lower()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            st = state[tid]
            st['class'] = cls
            st['positions'].append((cx, cy))
            if len(st['positions']) > 120:
                st['positions'] = st['positions'][-120:]

            # "Стоит"
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

            # --- Стоп-линия ---
            if cls in VEHICLE_CLASSES and len(st['positions']) >= 2:
                p_prev = st['positions'][-2]
                p_curr = st['positions'][-1]
                for idx, (A, B) in enumerate(zones['stop_lines']):
                    if idx in st['crossed_stop']:
                        continue
                    if segments_intersect(p_prev, p_curr, A, B):
                        st['crossed_stop'].add(idx)
                        st['crossed_stop_t'] = t_sec

                        # Учитываем стабильность светофора
                        time_since_change = (
                            t_sec - tl_last_change_t
                            if tl_last_change_t is not None else 999
                        )
                        # Нарушение ТОЛЬКО если:
                        #   1) светофор стабильно красный/жёлтый
                        #   2) с момента смены прошло >= tl_wait_after_change
                        if (tl_stable == 'red' and
                                time_since_change >= tl_wait_after_change):
                            st['last_violation_t'] = t_sec
                            st['violation_type'] = 'RED LIGHT'
                        elif (tl_stable == 'yellow' and
                                time_since_change >= tl_wait_after_change):
                            st['last_violation_t'] = t_sec
                            st['violation_type'] = 'YELLOW LIGHT'
                        # На зелёном / переходе — НЕ считаем

            # --- Сплошная ---
            if cls in VEHICLE_CLASSES and len(st['positions']) >= 2:
                p_prev = st['positions'][-2]
                p_curr = st['positions'][-1]
                for idx, (A, B) in enumerate(zones['solid_lines']):
                    if idx in st['crossed_solid']:
                        continue
                    if segments_intersect(p_prev, p_curr, A, B):
                        st['crossed_solid'].add(idx)
                        st['last_violation_t'] = t_sec
                        st['violation_type'] = 'SOLID LINE'

        # Сброс crossed_stop через 10 секунд (машина уехала)
        for tid, st in state.items():
            if st.get('crossed_stop_t') is not None and \
               (t_sec - st['crossed_stop_t']) > 10:
                st['crossed_stop'].clear()
                st['crossed_stop_t'] = None

        # ============================================================
        # Отрисовка
        # ============================================================

        # 1. Зоны
        overlay = frame.copy()
        for poly in zones['crosswalk_zones']:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
        for A, B in zones['stop_lines']:
            cv2.line(overlay, A, B, (0, 0, 255), 4)
        for A, B in zones['solid_lines']:
            cv2.line(overlay, A, B, (255, 0, 255), 4)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # 2. Треки
        for tr in tracks:
            x1, y1, x2, y2 = tr['bbox']
            cls = tr.get('class', '').lower()
            tid = tr['track_id']
            st = state[tid]
            is_violating = (st['last_violation_t'] is not None and
                            (t_sec - st['last_violation_t']) < violation_ttl)

            if cls in VEHICLE_CLASSES:
                if is_violating:
                    color = C_VIOLATION
                    label = f"#{tid} {cls} ⚠ {st['violation_type']}"
                    thick = 3
                elif (st['stationary_start_t'] is not None and
                      (t_sec - st['stationary_start_t']) >= stopped_dwell):
                    color = C_STOPPED
                    label = f"#{tid} {cls} STOPPED"
                    thick = 2
                else:
                    color = C_OK
                    label = f"#{tid} {cls}"
                    thick = 2
            elif cls in PED_CLASSES:
                # bbox пешехода пересекает любую зебру?
                in_cross = any(
                    _bbox_intersects_polygon([x1, y1, x2, y2], poly)
                    for poly in zones['crosswalk_zones']
                )
                color = C_PED_IN if in_cross else C_PED_OUT
                label = f"#{tid} {cls}" + ("" if in_cross else " JW")
                thick = 2
            elif cls in ('traffic light', 'traffic_light'):
                color = tl_color(tl_state)
                label = f"TL {tl_state}"
                thick = 4
            else:
                color = C_UNKNOWN
                label = f"#{tid} {cls}"
                thick = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # 3. Светофор из zones.json — рисуем ВСЕГДА (даже если YOLO не нашёл)
        for bbox in zones['traffic_lights']:
            x1, y1, x2, y2 = bbox
            c = tl_color(tl_state)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 4)
            cv2.putText(frame, f"TL {tl_state.upper()}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

        # 4. Плашка светофора сверху по центру
        c = tl_color(tl_state)
        text = f"TRAFFIC LIGHT: {tl_state.upper()}"
        if tl_stable != tl_state:
            text += f"  (stable: {tl_stable})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(frame,
                      (w // 2 - tw // 2 - 15, 8),
                      (w // 2 + tw // 2 + 15, 8 + th + 16),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (w // 2 - tw // 2 - 15, 8),
                      (w // 2 + tw // 2 + 15, 8 + th + 16),
                      c, 3)
        cv2.putText(frame, text,
                    (w // 2 - tw // 2, 8 + th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, c, 2, cv2.LINE_AA)

        # 5. Легенда снизу слева
        legend = [
            ("OK", C_OK),
            ("VIOLATION", C_VIOLATION),
            ("STOPPED", C_STOPPED),
            ("PED in crosswalk", C_PED_IN),
            ("PED jaywalking", C_PED_OUT),
        ]
        yl = h - 15 - len(legend) * 22
        for name, col in legend:
            cv2.rectangle(frame, (15, yl - 14), (35, yl + 2), col, -1)
            cv2.putText(frame, name, (45, yl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            yl += 22

        # 6. Время / кадр
        cv2.putText(frame, f"t={t_sec:5.1f}s  frame={frame_idx}",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx % 60 == 0:
            n_viol = sum(1 for s in state.values()
                         if s['last_violation_t'] is not None and
                         t_sec - s['last_violation_t'] < violation_ttl)
            print(f"  f{frame_idx} t={t_sec:.1f}s tracks={len(tracks)} "
                  f"TL={tl_state} stable={tl_stable} viol={n_viol}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n✅ Сохранено: {output_path}")


# =====================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        args = ['traffic_short.mp4']

    # Если последний аргумент оканчивается на .json — это путь к зонам
    zones_path = 'zones.json'
    if args and args[-1].endswith('.json'):
        zones_path = args[-1]
        args = args[:-1]

    for v in args:
        if not os.path.exists(v):
            print(f"⚠️  Пропускаю (нет файла): {v}")
            continue
        out = v.replace('.mp4', '_annotated.mp4')
        process_video(v, out, zones_path=zones_path)