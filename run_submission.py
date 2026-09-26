import json
import sys
from src.solution import run_submission

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_submission.py <video_path> [output_json]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    predictions = run_submission(video_path, output_path)
    print(json.dumps(predictions, indent=2))