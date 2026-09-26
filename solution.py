"""
solution.py — root-level entry point.

The full implementation lives in src/solution.py. This file re-exports
the required public interface so that reviewers can import from the
repository root directly:

    from solution import detect_events
    from solution import RiskEstimator

It also works as a CLI:

    python solution.py <video_path> [output_json]
"""

import sys
import json
from pathlib import Path

# Ensure the repo root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export public API
from src.solution import detect_events, run_submission, evaluate  # noqa: F401
from src.risk import RiskEstimator  # noqa: F401


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solution.py <video_path> [output_json]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    predictions = run_submission(video_path, output_path)
    print(json.dumps(predictions, indent=2))
