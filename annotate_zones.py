import cv2
import json
import sys
import os
import numpy as np


MODE_STOP_LINE = 1
MODE_CROSSWALK = 2
MODE_SOLID_LINE = 3
MODE_NOSTOP = 4
MODE_TRAFFIC_LIGHT = 5

LINE_MODES = {MODE_STOP_LINE, MODE_SOLID_LINE}
POLYGON_MODES = {MODE_CROSSWALK, MODE_NOSTOP}
BBOX_MODES = {MODE_TRAFFIC_LIGHT}

MODE_KEYS = {
    MODE_STOP_LINE: 'stop_lines',
    MODE_CROSSWALK: 'crosswalk_zones',
    MODE_SOLID_LINE: 'solid_lines',
    MODE_NOSTOP: 'no_stop_zones',
    MODE_TRAFFIC_LIGHT: 'traffic_lights',
}

COLORS = {
    MODE_STOP_LINE: (0, 0, 255),
    MODE_CROSSWALK: (0, 255, 255),
    MODE_SOLID_LINE: (255, 0, 255),
    MODE_NOSTOP: (0, 165, 255),
    MODE_TRAFFIC_LIGHT: (0, 255, 0),
}

NAMES = {
    MODE_STOP_LINE: 'STOP-LINE',
    MODE_CROSSWALK: 'CROSSWALK',
    MODE_SOLID_LINE: 'SOLID-LINE',
    MODE_NOSTOP: 'NO-STOP',
    MODE_TRAFFIC_LIGHT: 'TRAFFIC-LIGHT',
}


# ---------------------------------------------------------------------

def _normalize_loaded_zones(z):
    """Приводит любые сохранённые зоны к единому формату."""
    out = {
        'stop_lines': [],
        'crosswalk_zones': [],
        'solid_lines': [],
        'no_stop_zones': [],
        'traffic_lights': [],
    }
    # stop_lines — список из [[x1,y1],[x2,y2]]
    for l in z.get('stop_lines', []):
        if isinstance(l, list) and len(l) == 2 and \
           isinstance(l[0], (list, tuple)) and len(l[0]) == 2:
            out['stop_lines'].append([[int(l[0][0]), int(l[0][1])],
                                      [int(l[1][0]), int(l[1][1])]])
    # crosswalks — список полигонов
    for poly in z.get('crosswalk_zones', []):
        if isinstance(poly, list) and len(poly) >= 3:
            out['crosswalk_zones'].append(
                [[int(p[0]), int(p[1])] for p in poly if len(p) >= 2]
            )
    # solid_lines — как stop_lines
    for l in z.get('solid_lines', []):
        if isinstance(l, list) and len(l) == 2 and \
           isinstance(l[0], (list, tuple)) and len(l[0]) == 2:
            out['solid_lines'].append([[int(l[0][0]), int(l[0][1])],
                                       [int(l[1][0]), int(l[1][1])]])
    # no_stop_zones — полигоны
    for poly in z.get('no_stop_zones', []):
        if isinstance(poly, list) and len(poly) >= 3:
            out['no_stop_zones'].append(
                [[int(p[0]), int(p[1])] for p in poly if len(p) >= 2]
            )
    # traffic_lights — список bbox [x1,y1,x2,y2] или [{'bbox': ...}]
    for tl in z.get('traffic_lights', []):
        if isinstance(tl, dict) and 'bbox' in tl:
            b = tl['bbox']
            out['traffic_lights'].append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        elif isinstance(tl, list) and len(tl) == 4:
            out['traffic_lights'].append([int(tl[0]), int(tl[1]), int(tl[2]), int(tl[3])])
        elif isinstance(tl, list) and len(tl) == 1 and isinstance(tl[0], list):
            # [ [x1,y1,x2,y2] ]
            b = tl[0]
            if len(b) == 4:
                out['traffic_lights'].append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
    return out


# ---------------------------------------------------------------------

class ZonesAnnotator:
    def __init__(self, frame, existing_path=None):
        self.frame = frame
        self.img = frame.copy()
        self.h, self.w = frame.shape[:2]

        # Загружаем существующие, если есть
        self.zones = {
            'stop_lines': [],
            'crosswalk_zones': [],
            'solid_lines': [],
            'no_stop_zones': [],
            'traffic_lights': [],
        }
        if existing_path and os.path.exists(existing_path):
            try:
                with open(existing_path) as f:
                    z = json.load(f)
                self.zones = _normalize_loaded_zones(z)
                print(f"✅ Загружены существующие зоны из {existing_path}")
                for k, v in self.zones.items():
                    print(f"   {k}: {len(v)}")
            except Exception as e:
                print(f"⚠️  Не удалось загрузить {existing_path}: {e}")

        self.mode = MODE_STOP_LINE
        self.points = []

    # ------------------------------------------------------------------

    def redraw(self):
        self.img = self.frame.copy()

        for key, mode in [
            ('stop_lines', MODE_STOP_LINE),
            ('crosswalk_zones', MODE_CROSSWALK),
            ('solid_lines', MODE_SOLID_LINE),
            ('no_stop_zones', MODE_NOSTOP),
            ('traffic_lights', MODE_TRAFFIC_LIGHT),
        ]:
            color = COLORS[mode]
            for zone in self.zones[key]:
                try:
                    if mode in LINE_MODES:
                        # zone = [[x1,y1],[x2,y2]]
                        p1 = (int(zone[0][0]), int(zone[0][1]))
                        p2 = (int(zone[1][0]), int(zone[1][1]))
                        cv2.line(self.img, p1, p2, color, 3)
                        cv2.circle(self.img, p1, 5, color, -1)
                        cv2.circle(self.img, p2, 5, color, -1)
                        cv2.putText(self.img, NAMES[mode],
                                    (p1[0], max(15, p1[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    elif mode in POLYGON_MODES:
                        # zone = [[x,y], ...]
                        pts = np.array(zone, dtype=np.int32)
                        if len(pts) >= 3:
                            cv2.polylines(self.img, [pts], True, color, 3)
                            overlay = self.img.copy()
                            cv2.fillPoly(overlay, [pts], color)
                            self.img = cv2.addWeighted(
                                overlay, 0.25, self.img, 0.75, 0)
                            for p in zone:
                                cv2.circle(self.img, (int(p[0]), int(p[1])), 4,
                                           color, -1)
                            cv2.putText(self.img, NAMES[mode],
                                        (int(zone[0][0]),
                                         max(15, int(zone[0][1]) - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    elif mode in BBOX_MODES:
                        # zone = [x1, y1, x2, y2]
                        x1, y1, x2, y2 = (int(zone[0]), int(zone[1]),
                                          int(zone[2]), int(zone[3]))
                        cv2.rectangle(self.img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(self.img, NAMES[mode],
                                    (x1, max(15, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    print(f"⚠️  Ошибка отрисовки {key}: {e} / zone={zone}")
                    continue

        # Текущие клики
        color = COLORS[self.mode]
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.img, (x, y), 6, color, -1)
            if i > 0:
                cv2.line(self.img, self.points[i - 1], (x, y), color, 2)
        # Замыкание полигона-превью
        if self.mode in POLYGON_MODES and len(self.points) >= 3:
            cv2.line(self.img, self.points[-1], self.points[0], color, 1)
        # Превью bbox
        if self.mode in BBOX_MODES and len(self.points) >= 2:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            cv2.rectangle(self.img, (min(xs), min(ys)),
                          (max(xs), max(ys)), color, 2)

        # Подсказка
        hint = (f"MODE: {NAMES[self.mode]}  pts: {len(self.points)}  |  "
                f"1-stop 2-cross 3-solid 4-nostop 5-TL  Enter-fin R-undo S-save Q-quit")
        cv2.putText(self.img, hint, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        y = 55
        for key in ['stop_lines', 'crosswalk_zones', 'solid_lines',
                    'no_stop_zones', 'traffic_lights']:
            cv2.putText(self.img, f"{key}: {len(self.zones[key])}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            y += 20

    # ------------------------------------------------------------------

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if self.mode in LINE_MODES and len(self.points) == 2:
                self._commit_line()
            elif self.mode in BBOX_MODES and len(self.points) == 4:
                self._commit_bbox()
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.redraw()

    def _commit_line(self):
        key = MODE_KEYS[self.mode]
        self.zones[key].append([
            [int(self.points[0][0]), int(self.points[0][1])],
            [int(self.points[1][0]), int(self.points[1][1])],
        ])
        self.points = []

    def _commit_bbox(self):
        key = MODE_KEYS[self.mode]
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        # Единый формат для traffic_lights: [x1,y1,x2,y2]
        self.zones[key].append(bbox)
        self.points = []

    def commit_polygon(self):
        if self.mode not in POLYGON_MODES:
            return
        if len(self.points) < 3:
            print(f"⚠️  Нужно 3+ точек (сейчас {len(self.points)})")
            return
        key = MODE_KEYS[self.mode]
        self.zones[key].append([[int(x), int(y)] for x, y in self.points])
        self.points = []
        self.redraw()

    def undo_last(self):
        if self.points:
            self.points = []
            self.redraw()
            return
        for key in ['stop_lines', 'crosswalk_zones', 'solid_lines',
                    'no_stop_zones', 'traffic_lights']:
            if self.zones[key]:
                self.zones[key].pop()
                break
        self.redraw()

    def set_mode(self, mode):
        # Если завершён полигон — сохраним
        if self.mode in POLYGON_MODES and len(self.points) >= 3:
            self.commit_polygon()
        self.mode = mode
        self.points = []
        self.redraw()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.zones, f, indent=2)
        print(f"\n✅ Сохранено в {path}")
        for k, v in self.zones.items():
            print(f"  {k}: {len(v)}")


# ---------------------------------------------------------------------

def main(video_path, output_json='zones.json'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не открыть {video_path}")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Не прочитать первый кадр")
        return

    h, w = frame.shape[:2]
    print(f"✅ {video_path}: {w}x{h}")
    print()
    print("Режимы:")
    print("  1 — стоп-линия    (2 pts)")
    print("  2 — зебра         (N pts + Enter)")
    print("  3 — сплошная      (2 pts)")
    print("  4 — no-stop zone  (N pts + Enter)")
    print("  5 — светофор      (4 pts, bbox)")
    print()
    print("ЛКМ — поставить точку, ПКМ — отменить, Enter — завершить, R — отменить фигуру")
    print("S — сохранить, Q — выход")
    print()

    ann = ZonesAnnotator(frame, existing_path=output_json)

    win = 'Annotate Zones'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, ann.mouse_cb)
    ann.redraw()

    while True:
        cv2.imshow(win, ann.img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("Выход без сохранения")
            break
        elif key == ord('s'):
            if ann.mode in POLYGON_MODES and len(ann.points) >= 3:
                ann.commit_polygon()
            ann.save(output_json)
            break
        elif key == ord('r'):
            ann.undo_last()
        elif key == 13:  # Enter
            ann.commit_polygon()
        elif key == ord('1'):
            ann.set_mode(MODE_STOP_LINE)
        elif key == ord('2'):
            ann.set_mode(MODE_CROSSWALK)
        elif key == ord('3'):
            ann.set_mode(MODE_SOLID_LINE)
        elif key == ord('4'):
            ann.set_mode(MODE_NOSTOP)
        elif key == ord('5'):
            ann.set_mode(MODE_TRAFFIC_LIGHT)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python annotate_zones.py <video.mp4> [zones.json]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'zones.json')