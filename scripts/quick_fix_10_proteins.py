#!/usr/bin/env python3
"""
QUICK FIX: Re-process 10 proteins with correct labels
"""
import os
import torch
from tqdm import tqdm
import sys
sys.path.append('./src/models')
from simple_builder_fixed import SimpleBuilderFixed

def fix_10_proteins():
    """Fix 10 proteins with correct labels"""
    builder = SimpleBuilderFixed(cutoff=8.0)
    
    # Get list of processed proteins
    processed_dir = './data/processed/emergency_200/'
    files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')][:10]
    
    print(f"Fixing labels for {len(files)} proteins...")
    
    fixed_graphs = []
    output_dir = './data/processed/phase2c_fixed_10'
    os.makedirs(output_dir, exist_ok=True)
    
    for f in tqdm(files, desc="Fixing"):
        protein_id = f.replace('_graph.pt', '')
        
        # Find original files
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        
        # Find ligand file
        ligand_paths = [
            f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2",
            f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf",
            f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.pdb"
        ]
        
        ligand_path = None
        for lp in ligand_paths:
            if os.path.exists(lp):
                ligand_path = lp
                break
        
        if os.path.exists(protein_path):
            graph = builder.build_from_pdb(protein_path, ligand_path, protein_id)
            if graph is not None:
                fixed_graphs.append(graph)
                torch.save(graph, f"{output_dir}/{protein_id}_fixed.pt")
                
                # Check label statistics
                num_pos = graph.y.sum().item()
                total = graph.y.shape[0]
                print(f"  {protein_id}: {num_pos}/{total} pocket residues ({num_pos/total*100:.1f}%)")
    
    print(f"\nFixed {len(fixed_graphs)} proteins")
    print(f"Saved to: {output_dir}/")
    
    return fixed_graphs

if __name__ == "__main__":
    fix_10_proteins()
