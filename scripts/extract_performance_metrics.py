import json
import os
import pandas as pd

def extract_metrics():
    metrics = {}
    
    # Check all possible JSON files
    json_files = [
        './experiments/results/phase2b/week2/training_fixed/hamgnn_performance_fixed.json',
        './experiments/results/phase2b/week2/training_fixed/training_history.json',
        './experiments/results/phase2c_final_training/final_results.json',
        './experiments/results/phase2c_final_training/training_history.json',
        './experiments/results/phase2_working/training_history.json',
    ]
    
    for file_path in json_files:
        if os.path.exists(file_path):
            print(f"\n=== Checking {file_path} ===")
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract any performance metrics
                if isinstance(data, dict):
                    for key, value in data.items():
                        if any(metric in key.lower() for metric in ['f1', 'precision', 'recall', 'accuracy', 'auc', 'loss']):
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value}")
                                metrics[key] = value
                        elif isinstance(value, list) and len(value) > 0:
                            # Check if list contains metrics
                            if all(isinstance(x, (int, float)) for x in value[:3]):
                                print(f"  {key}: list with {len(value)} values")
                                metrics[f"{key}_mean"] = sum(value)/len(value)
                                metrics[f"{key}_max"] = max(value)
                                metrics[f"{key}_min"] = min(value)
                
            except Exception as e:
                print(f"  Error reading: {e}")
    
    # Save extracted metrics
    if metrics:
        with open('./experiments/results/performance_metrics_summary.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Saved metrics to performance_metrics_summary.json")
        
        # Also create a simple summary
        print("\n=== PERFORMANCE SUMMARY ===")
        for key in sorted(metrics.keys()):
            if any(term in key.lower() for term in ['f1', 'precision', 'recall']):
                print(f"{key}: {metrics[key]}")
    
    return metrics

if __name__ == "__main__":
    extract_metrics()
