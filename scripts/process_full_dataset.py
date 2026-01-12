#!/usr/bin/env python3
"""
Process ALL 1,182 proteins with physics-enhanced features
"""
import os
import sys
import torch
from tqdm import tqdm
import time
import json

# Add src to path
sys.path.append('./src/models')
from physics_enhanced_builder import PhysicsEnhancedGraphBuilder

def process_full_dataset():
    """Process all proteins in the dataset"""
    print("=" * 60)
    print("PHASE 2C: PROCESSING FULL DATASET (1,182 proteins)")
    print("=" * 60)
    
    # Initialize builder
    builder = PhysicsEnhancedGraphBuilder(cutoff_distance=8.0)
    
    # Read splits from Phase 1
    splits = {
        'train': './experiments/results/phase1/splits/train_ids.txt',
        'val': './experiments/results/phase1/splits/val_ids.txt',
        'test': './experiments/results/phase1/splits/test_ids.txt'
    }
    
    stats = {
        'total_processed': 0,
        'failed': 0,
        'processing_times': [],
        'graph_sizes': [],
        'memory_usage': []
    }
    
    start_time = time.time()
    
    for split_name, split_file in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Read protein IDs
        with open(split_file, 'r') as f:
            protein_ids = [line.strip() for line in f if line.strip()]
        
        print(f"  Found {len(protein_ids)} proteins")
        
        processed_count = 0
        failed_ids = []
        
        for protein_id in tqdm(protein_ids, desc=f"Processing {split_name}"):
            try:
                # Construct file paths
                protein_dir = f"./data/PDBbind/refined-set/{protein_id}"
                protein_path = f"{protein_dir}/{protein_id}_protein.pdb"
                
                # Try different ligand formats
                ligand_paths = [
                    f"{protein_dir}/{protein_id}_ligand.mol2",
                    f"{protein_dir}/{protein_id}_ligand.sdf",
                    f"{protein_dir}/{protein_id}_ligand.pdb"
                ]
                
                ligand_path = None
                for lp in ligand_paths:
                    if os.path.exists(lp):
                        ligand_path = lp
                        break
                
                if not os.path.exists(protein_path):
                    print(f"  Warning: Missing protein file for {protein_id}")
                    stats['failed'] += 1
                    failed_ids.append(protein_id)
                    continue
                
                # Build graph
                split_start = time.time()
                graph = builder.build_from_pdb(protein_path, ligand_path, protein_id)
                split_time = time.time() - split_start
                
                if graph is not None:
                    # Save graph
                    output_dir = f"./data/processed/physics_graphs/full/{split_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = f"{output_dir}/{protein_id}_graph.pt"
                    torch.save(graph, output_path)
                    
                    # Update statistics
                    stats['total_processed'] += 1
                    stats['processing_times'].append(split_time)
                    stats['graph_sizes'].append(graph.num_nodes)
                    processed_count += 1
                    
                    # Save every 10 proteins (checkpoint)
                    if processed_count % 10 == 0:
                        print(f"  Processed {processed_count}/{len(protein_ids)}")
                else:
                    stats['failed'] += 1
                    failed_ids.append(protein_id)
                    
            except Exception as e:
                print(f"  Error processing {protein_id}: {e}")
                stats['failed'] += 1
                failed_ids.append(protein_id)
        
        print(f"  {split_name}: {processed_count} succeeded, {len(failed_ids)} failed")
        
        # Save failed IDs for debugging
        if failed_ids:
            failed_file = f"./experiments/results/phase2c/week1/failed_{split_name}.txt"
            with open(failed_file, 'w') as f:
                for fid in failed_ids:
                    f.write(f"{fid}\n")
    
    # Calculate statistics
    total_time = time.time() - start_time
    stats['total_time_hours'] = total_time / 3600
    stats['avg_time_per_protein'] = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0
    stats['success_rate'] = stats['total_processed'] / (stats['total_processed'] + stats['failed'])
    
    # Save statistics
    stats_file = "./experiments/results/phase2c/week1/scaling_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total time: {stats['total_time_hours']:.1f} hours")
    print(f"  Avg time per protein: {stats['avg_time_per_protein']:.2f}s")
    print("=" * 60)
    
    return stats

if __name__ == "__main__":
    process_full_dataset()
