#!/usr/bin/env python3
"""
Fix torch.load issue by adding safe globals or using weights_only=False
"""
import torch
from torch_geometric.data import Data
import glob
import os
import sys

# Add Data to safe globals for weights_only loading
torch.serialization.add_safe_globals([Data])

def fix_and_verify():
    """Fix loading issue and verify all graphs"""
    print("Fixing torch.load issue...")
    
    # First, let's verify the issue
    test_file = "./data/processed/phase2c_final_300/train/"
    files = os.listdir(test_file)
    if not files:
        print("No files found!")
        return False
    
    test_path = os.path.join(test_file, files[0])
    print(f"Testing with: {test_path}")
    
    # Try with weights_only=True after adding safe globals
    try:
        graph = torch.load(test_path, weights_only=True)
        print("✓ Success with weights_only=True after add_safe_globals")
        print(f"  Graph has {graph.num_nodes} nodes, {graph.x.shape[1]} features")
        return True
    except Exception as e:
        print(f"✗ Still failing: {e}")
        print("Trying weights_only=False (less secure but works)...")
        
        # Try weights_only=False
        try:
            graph = torch.load(test_path, weights_only=False)
            print("✓ Success with weights_only=False")
            print(f"  Graph has {graph.num_nodes} nodes, {graph.x.shape[1]} features")
            
            # Save all graphs with proper format
            print("\nConverting all graphs to weights_only=True compatible format...")
            convert_all_graphs()
            return True
        except Exception as e2:
            print(f"✗ Complete failure: {e2}")
            return False

def convert_all_graphs():
    """Convert all graphs to a weights_only=True compatible format"""
    import pickle
    
    # Create new directory for converted graphs
    new_dir = "./data/processed/phase2c_final_300_converted"
    os.makedirs(f"{new_dir}/train", exist_ok=True)
    os.makedirs(f"{new_dir}/val", exist_ok=True)
    os.makedirs(f"{new_dir}/test", exist_ok=True)
    
    converted = 0
    failed = 0
    
    for split in ['train', 'val', 'test']:
        src_dir = f"./data/processed/phase2c_final_300/{split}"
        dst_dir = f"{new_dir}/{split}"
        
        if not os.path.exists(src_dir):
            continue
            
        print(f"\nConverting {split} graphs...")
        
        for fname in os.listdir(src_dir):
            if fname.endswith('.pt'):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                
                try:
                    # Load with weights_only=False
                    graph = torch.load(src_path, weights_only=False)
                    
                    # Convert to dictionary (weights_only compatible)
                    graph_dict = {
                        'x': graph.x,
                        'edge_index': graph.edge_index,
                        'edge_attr': graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                        'y': graph.y,
                        'pos': graph.pos if hasattr(graph, 'pos') else None,
                        'protein_id': graph.protein_id if hasattr(graph, 'protein_id') else fname.replace('_graph.pt', ''),
                        'num_nodes': graph.num_nodes
                    }
                    
                    # Save as dictionary
                    torch.save(graph_dict, dst_path)
                    
                    # Verify it loads with weights_only=True
                    test_load = torch.load(dst_path, weights_only=True)
                    converted += 1
                    
                except Exception as e:
                    print(f"  Failed {fname}: {e}")
                    failed += 1
    
    print(f"\nConversion complete: {converted} converted, {failed} failed")
    
    # Test one converted file
    if converted > 0:
        test_file = f"{new_dir}/train/{os.listdir(f'{new_dir}/train')[0]}"
        test_dict = torch.load(test_file, weights_only=True)
        print(f"\nTest load successful: {test_dict['protein_id']} with {test_dict['num_nodes']} nodes")
        
        # Update symlink or copy
        if os.path.exists("./data/processed/phase2c_final_300"):
            os.rename("./data/processed/phase2c_final_300", 
                     "./data/processed/phase2c_final_300_original")
        os.symlink(new_dir, "./data/processed/phase2c_final_300")
        print(f"Updated symlink to use converted graphs")

def check_class_distribution():
    """Check class distribution in converted graphs"""
    print("\nChecking class distribution...")
    
    try:
        import torch
        import numpy as np
        
        pos_counts = []
        neg_counts = []
        
        train_dir = "./data/processed/phase2c_final_300/train"
        
        if not os.path.exists(train_dir):
            print(f"Directory {train_dir} does not exist!")
            return
        
        files = os.listdir(train_dir)[:10]  # Check first 10 files
        
        for fname in files:
            if fname.endswith('.pt'):
                try:
                    graph_dict = torch.load(os.path.join(train_dir, fname), weights_only=True)
                    labels = graph_dict['y'].numpy()
                    pos_counts.append(np.sum(labels))
                    neg_counts.append(len(labels) - np.sum(labels))
                except Exception as e:
                    print(f"  Error loading {fname}: {e}")
        
        if pos_counts:
            total_pos = sum(pos_counts)
            total_neg = sum(neg_counts)
            total = total_pos + total_neg
            
            print(f"Sample of {len(pos_counts)} graphs:")
            print(f"Total residues: {total:,}")
            print(f"Positive (pocket): {total_pos:,} ({total_pos/total:.3%})")
            print(f"Negative (non-pocket): {total_neg:,} ({total_neg/total:.3%})")
            print(f"Optimal pos_weight = neg/pos = {total_neg/total_pos:.2f}")
        else:
            print("No valid graphs found!")
            
    except Exception as e:
        print(f"Error in distribution check: {e}")

if __name__ == "__main__":
    # First, try to fix the loading issue
    success = fix_and_verify()
    
    if success:
        # Check class distribution
        check_class_distribution()
        
        print("\n" + "="*60)
        print("TORCH.LOAD ISSUE FIXED!")
        print("Now you can run training with weights_only=True")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("FIX FAILED - Need alternative approach")
        print("="*60)
