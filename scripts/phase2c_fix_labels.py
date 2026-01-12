#!/usr/bin/env python3
"""
FIX PHASE 2C: Use PROVEN builder from Phase 2B to get correct labels
"""
import os
import torch
import json
from tqdm import tqdm
import sys
sys.path.append('./src/models')

# Import the PROVEN builder from Phase 2B
from simple_enhanced_builder import SimpleEnhancedGraphBuilder

def reprocess_with_correct_builder():
    """Reprocess 50 proteins with CORRECT builder"""
    print("Reprocessing with PROVEN Phase 2B builder...")
    
    builder = SimpleEnhancedGraphBuilder(cutoff_distance=8.0)
    
    # Read train IDs
    with open('./experiments/results/phase1/splits/train_ids.txt', 'r') as f:
        all_ids = [line.strip() for line in f if line.strip()]
    
    # Take first 50
    target_ids = all_ids[:50]
    print(f"Reprocessing {len(target_ids)} proteins with correct builder...")
    
    processed = []
    output_dir = './data/processed/phase2c_correct_labels'
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for pid in tqdm(target_ids, desc="Reprocessing"):
        protein_path = f"./data/PDBbind/refined-set/{pid}/{pid}_protein.pdb"
        
        # Try different ligand formats
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
                # Check if graph has positive labels
                positive_count = graph.y.sum().item()
                if positive_count > 0:  # Only keep if has positive labels
                    processed.append(graph)
                    torch.save(graph, f"{output_dir}/{pid}_graph.pt")
                    success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(target_ids)} proteins WITH POSITIVE LABELS")
    
    # Check class balance
    total_nodes = 0
    total_positives = 0
    for graph in processed:
        total_nodes += graph.y.shape[0]
        total_positives += graph.y.sum().item()
    
    if total_nodes > 0:
        print(f"Class balance: {total_positives/total_nodes:.3%} positive ({total_positives}/{total_nodes})")
    
    return processed

def check_existing_labels():
    """Check labels in existing emergency graphs"""
    print("\n=== CHECKING EXISTING EMERGENCY GRAPHS ===")
    graph_dir = './data/processed/emergency_200/'
    
    total_positive = 0
    total_nodes = 0
    graphs_with_positive = 0
    
    files = os.listdir(graph_dir)[:10]  # Check first 10
    for f in files:
        if f.endswith('.pt'):
            try:
                graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
                positives = graph.y.sum().item()
                total_nodes += graph.y.shape[0]
                total_positive += positives
                if positives > 0:
                    graphs_with_positive += 1
            except:
                pass
    
    print(f"Checked {len(files)} emergency graphs:")
    print(f"  Graphs with positive labels: {graphs_with_positive}/{len(files)}")
    print(f"  Total positives: {total_positive}/{total_nodes} ({total_positive/total_nodes if total_nodes>0 else 0:.3%})")

if __name__ == "__main__":
    check_existing_labels()
    print("\n" + "="*60)
    reprocess_with_correct_builder()
