import json
import sys
from src.solution import run_submission, evaluate

def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <predictions_json> <ground_truth_json> [output_metrics]")
        sys.exit(1)
    
    predictions_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    metrics = evaluate(predictions_path, ground_truth_path)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()