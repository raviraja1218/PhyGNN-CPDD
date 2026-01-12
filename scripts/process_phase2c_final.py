#!/usr/bin/env python3
"""
Process 300 final proteins with physics features for Phase 2C completion
"""
import os
import sys
import torch
from tqdm import tqdm
import time
import json

sys.path.append('./src/models')
from physics_builder_final import PhysicsEnhancedGraphBuilder

def process_final_dataset():
    """Process 300 proteins for final Phase 2C"""
    print("=" * 60)
    print("PHASE 2C FINAL: PROCESSING 300 PROTEINS")
    print("=" * 60)
    
    # Initialize builder
    builder = PhysicsEnhancedGraphBuilder(cutoff_distance=8.0)
    
    # Read protein selections
    splits = {
        'train': './experiments/results/phase2c_final/train_240.txt',
        'val': './experiments/results/phase2c_final/val_30.txt',
        'test': './experiments/results/phase2c_final/test_30.txt'
    }
    
    stats = {'total': 0, 'success': 0, 'failed': 0, 'times': []}
    
    for split_name, split_file in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        with open(split_file, 'r') as f:
            protein_ids = [line.strip() for line in f if line.strip()]
        
        for protein_id in tqdm(protein_ids, desc=split_name):
            try:
                # Find protein files
                protein_dir = f"./data/PDBbind/refined-set/{protein_id}"
                protein_path = f"{protein_dir}/{protein_id}_protein.pdb"
                
                # Find ligand file
                ligand_path = None
                for ext in ['.mol2', '.sdf', '.pdb']:
                    test_path = f"{protein_dir}/{protein_id}_ligand{ext}"
                    if os.path.exists(test_path):
                        ligand_path = test_path
                        break
                
                if not os.path.exists(protein_path):
                    stats['failed'] += 1
                    continue
                
                # Build graph
                start_time = time.time()
                graph = builder.build_from_pdb(protein_path, ligand_path, protein_id)
                build_time = time.time() - start_time
                
                if graph is not None:
                    # Save graph
                    output_dir = f"./data/processed/phase2c_final_300/{split_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(graph, f"{output_dir}/{protein_id}_graph.pt")
                    
                    stats['success'] += 1
                    stats['total'] += 1
                    stats['times'].append(build_time)
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"Error with {protein_id}: {e}")
                stats['failed'] += 1
    
    # Save statistics
    stats['avg_time'] = sum(stats['times'])/len(stats['times']) if stats['times'] else 0
    stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
    
    with open('./experiments/results/phase2c_final/processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Success: {stats['success']}/{stats['total']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average time per protein: {stats['avg_time']:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    process_final_dataset()
