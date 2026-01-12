import os
from src.models.graph_builder import GraphBuilder
from tqdm import tqdm
import torch

def process_all_proteins():
    """Process all proteins in the dataset"""
    builder = GraphBuilder(cutoff_distance=8.0)
    
    # Read train IDs from Phase 1
    with open('./experiments/results/phase1/splits/train_ids.txt', 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(train_ids)} training proteins...")
    
    processed_count = 0
    failed_count = 0
    
    for protein_id in tqdm(train_ids, desc="Processing proteins"):
        # Construct file paths
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2"
        
        if not os.path.exists(protein_path):
            # Try alternative ligand format
            ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf"
        
        if os.path.exists(protein_path):
            # Build graph
            graph = builder.build_from_pdb(protein_path, ligand_path, protein_id)
            
            if graph is not None:
                # Save graph
                output_path = f"./data/processed/graphs/train/{protein_id}_graph.pt"
                torch.save(graph, output_path)
                processed_count += 1
            else:
                failed_count += 1
                print(f"Failed to process {protein_id}")
        else:
            failed_count += 1
            print(f"Missing protein file: {protein_id}")
    
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")

if __name__ == "__main__":
    process_all_proteins()
