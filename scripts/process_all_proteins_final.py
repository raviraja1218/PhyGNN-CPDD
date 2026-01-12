#!/usr/bin/env python3
"""
FINAL: Process ALL 1,182 proteins for Phase 2C completion
"""
import os
import torch
import sys
import time
import json
from tqdm import tqdm

sys.path.append('./src/models')
from simple_builder_fixed import SimpleBuilderFixed

def process_all_proteins():
    """Process entire PDBbind dataset"""
    builder = SimpleBuilderFixed(cutoff=8.0)
    
    # All splits
    splits = {
        'train': './experiments/results/phase1/splits/train_ids.txt',
        'val': './experiments/results/phase1/splits/val_ids.txt',
        'test': './experiments/results/phase1/splits/test_ids.txt'
    }
    
    total_start = time.time()
    stats = {'total_processed': 0, 'failed': 0}
    
    for split_name, split_file in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        with open(split_file, 'r') as f:
            protein_ids = [line.strip() for line in f if line.strip()]
        
        split_dir = f"./data/processed/physics_graphs_full/{split_name}"
        os.makedirs(split_dir, exist_ok=True)
        
        split_processed = 0
        split_failed = 0
        
        for pid in tqdm(protein_ids, desc=f"  {split_name}"):
            protein_path = f"./data/PDBbind/refined-set/{pid}/{pid}_protein.pdb"
            
            # Find ligand
            ligand_paths = [
                f"./data/PDBbind/refined-set/{pid}/{pid}_ligand.mol2",
                f"./data/PDBbind/refined-set/{pid}/{pid}_ligand.sdf",
                f"./data/PDBbind/refined-set/{pid}/{pid}_ligand.pdb"
            ]
            
            ligand_path = None
            for lp in ligand_paths:
                if os.path.exists(lp):
                    ligand_path = lp
                    break
            
            if os.path.exists(protein_path):
                graph = builder.build_from_pdb(protein_path, ligand_path, pid)
                if graph is not None:
                    torch.save(graph, f"{split_dir}/{pid}_graph.pt")
                    split_processed += 1
                    stats['total_processed'] += 1
                else:
                    split_failed += 1
                    stats['failed'] += 1
            else:
                split_failed += 1
                stats['failed'] += 1
        
        print(f"  {split_name}: {split_processed} succeeded, {split_failed} failed")
    
    total_time = time.time() - total_start
    stats['total_time_hours'] = total_time / 3600
    stats['success_rate'] = stats['total_processed'] / (stats['total_processed'] + stats['failed'])
    
    # Save statistics
    os.makedirs('./experiments/results/phase2c/final_processing', exist_ok=True)
    with open('./experiments/results/phase2c/final_processing/processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Total time: {stats['total_time_hours']:.1f} hours")
    print(f"{'='*60}")
    
    return stats

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2C FINAL: PROCESSING ALL 1,182 PROTEINS")
    print("Estimated time: 24 hours")
    print("=" * 60)
    process_all_proteins()
