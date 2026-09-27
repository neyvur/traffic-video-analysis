import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.annotate import process_video_annotated  # noqa: E402

STATIC_DIR = ROOT / 'api' / 'static'
STATIC_DIR.mkdir(exist_ok=True)


EVENT_MAP = {
    'accident':             'accident',
    'near_miss':            'near_miss',
    'red_light':            'red_light',
    'wrong_way':            'wrong_way',
    'illegal_u_turn':       'illegal_u_turn',
    'illegal_turn':         'illegal_turn',
    'stopped_vehicle':      'stopped_vehicle',
    'jaywalking':           'jaywalking',
    'failure_to_yield':     'failure_to_yield',
    'crossing_solid_lines': 'solid_line_crossing',
    'stop_line_violation':  'stop_line',
    'congestion':           'congestion',
    'obstacle_on_road':     'road_obstacle',
    'fire_smoke':           'fire_smoke',
}


app = FastAPI(title='Traffic Video Analysis API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://wiut-cv-site.vercel.app",
        "https://wiut-cv-site-mnyjxya8d-gama-core.vercel.app",
    ],
    allow_origin_regex=r"https://wiut-cv-site.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/')
def root():
    return {'status': 'ok'}


@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.get('/api/samples')
def samples():
    return [
        {
            "id": f"sample-{n}",
            "title": f"Sample 0{n}",
            "videoUrl": None,
            "duration": "60s",
            "resolution": "1280x720",
            "fps": "30",
            "events": [],
            "annotatedVideoUrl": None,
            "mock": True,
        }
        for n in (1, 2, 3)
    ]


@app.post('/api/analyze')
def analyze(video: UploadFile = File(...)):
    """Sync endpoint — не блокирует event loop."""
    if not video.filename:
        raise HTTPException(400, 'No filename')

    suffix = os.path.splitext(video.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            shutil.copyfileobj(video.file, tmp)
        finally:
            video.file.close()
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex
    annotated_path = STATIC_DIR / f'{job_id}.mp4'

    try:
        zones_path = str(ROOT / 'zones.json')

        events, risk_scores = process_video_annotated(
            tmp_path,
            str(annotated_path),
            zones_path=zones_path,
        )

        import cv2
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        site_events = []
        for s, e, et in events:
            label = EVENT_MAP.get(et)
            if label is None:
                continue
            site_events.append({
                'start': float(s), 'end': float(e), 'label': label,
            })

        return {
            'events': site_events,
            'risk': [
                {'time': float(t), 'score': float(r)}
                for t, r in risk_scores
            ],
            'meta': {
                'durationSec': round(n / fps, 2) if fps else None,
                'fps': round(fps, 2),
                'resolution': f'{w}x{h}',
            },
            'annotatedVideoUrl': f'/static/{job_id}.mp4',
            'mock': False,
        }

    except Exception as e:
        raise HTTPException(500, f'Analysis failed: {e}')

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass