#!/usr/bin/env python3
"""
Create dataset splits (train/val/test)
Ensures no data leakage
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_splits():
    """Create 70/15/15 train/val/test splits"""
    
    # Load valid complexes
    results_dir = Path("./experiments/results/phase1")
    complexes_file = results_dir / "valid_complexes.csv"
    
    if not complexes_file.exists():
        print("Error: valid_complexes.csv not found. Run validate_dataset.py first.")
        return
    
    df = pd.read_csv(complexes_file)
    pdb_ids = df['pdb_id'].tolist()
    
    print(f"Creating splits for {len(pdb_ids)} complexes")
    
    # Shuffle IDs
    np.random.seed(42)  # For reproducibility
    shuffled_ids = np.random.permutation(pdb_ids)
    
    # Split indices
    n = len(shuffled_ids)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    
    train_ids = shuffled_ids[:train_end].tolist()
    val_ids = shuffled_ids[train_end:val_end].tolist()
    test_ids = shuffled_ids[val_end:].tolist()
    
    print(f"Train: {len(train_ids)} complexes")
    print(f"Val: {len(val_ids)} complexes")
    print(f"Test: {len(test_ids)} complexes")
    
    # Save splits
    splits_dir = results_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / "train_ids.txt", 'w') as f:
        f.write("\n".join(train_ids))
    
    with open(splits_dir / "val_ids.txt", 'w') as f:
        f.write("\n".join(val_ids))
    
    with open(splits_dir / "test_ids.txt", 'w') as f:
        f.write("\n".join(test_ids))
    
    # Save split info as JSON
    split_info = {
        "total_complexes": n,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "train_percentage": 70.0,
        "val_percentage": 15.0,
        "test_percentage": 15.0,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids
    }
    
    with open(splits_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Verify no overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    assert len(train_set.intersection(val_set)) == 0, "Train-Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train-Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val-Test overlap!"
    
    print("\n✅ Splits created successfully!")
    print(f"Saved to: {splits_dir}/")
    
    return split_info

def main():
    print("=== Creating Dataset Splits ===")
    splits = create_splits()
    
    if splits:
        print("\n=== SPLIT SUMMARY ===")
        print(f"Total complexes: {splits['total_complexes']}")
        print(f"Train: {splits['train_count']} ({splits['train_percentage']}%)")
        print(f"Validation: {splits['val_count']} ({splits['val_percentage']}%)")
        print(f"Test: {splits['test_count']} ({splits['test_percentage']}%)")
        
        # Verify counts
        expected_total = int(splits['total_complexes'] * 0.7) + \
                        int(splits['total_complexes'] * 0.15) + \
                        (splits['total_complexes'] - int(splits['total_complexes'] * 0.7) - int(splits['total_complexes'] * 0.15))
        
        print(f"\nVerification:")
        print(f"  Expected: {splits['total_complexes']}, Actual: {expected_total}")
        print(f"  Match: {'✅' if splits['total_complexes'] == expected_total else '❌'}")

if __name__ == "__main__":
    main()
