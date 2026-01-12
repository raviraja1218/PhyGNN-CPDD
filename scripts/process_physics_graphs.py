"""
Process all proteins with physics-enhanced graph builder
"""
import os
import torch
from src.models.physics_enhanced_builder import PhysicsEnhancedGraphBuilder
from tqdm import tqdm
import json

def process_physics_graphs():
    """Process all proteins with physics features"""
    builder = PhysicsEnhancedGraphBuilder(cutoff_distance=8.0)
    
    # Read train IDs from Phase 2A (same 70 proteins for consistency)
    with open('./experiments/results/phase2_working/processed_ids.txt', 'r') as f:
        protein_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(protein_ids)} proteins with physics features...")
    
    processed = []
    failed = []
    statistics = []
    
    for protein_id in tqdm(protein_ids, desc="Processing"):
        # Construct file paths
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2"
        
        if not os.path.exists(ligand_path):
            # Try alternative format
            ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf"
        
        if os.path.exists(protein_path):
            try:
                # Build physics-enhanced graph
                graph = builder.build_from_pdb(protein_path, ligand_path, protein_id)
                
                if graph is not None:
                    # Save graph
                    output_path = f"./data/processed/physics_graphs/train/{protein_id}_physics.pt"
                    torch.save(graph, output_path)
                    
                    # Record statistics
                    stats = {
                        'protein_id': protein_id,
                        'num_nodes': graph.num_nodes,
                        'num_edges': graph.edge_index.shape[1],
                        'node_feat_dim': graph.x.shape[1],
                        'edge_physics_dim': graph.edge_physics.shape[1] if hasattr(graph, 'edge_physics') else 0,
                        'pocket_residues': graph.y.sum().item(),
                        'total_residues': graph.num_nodes
                    }
                    statistics.append(stats)
                    processed.append(protein_id)
                else:
                    failed.append(protein_id)
                    print(f"✗ Failed to build physics graph for {protein_id}")
                    
            except Exception as e:
                failed.append(protein_id)
                print(f"✗ Error processing {protein_id}: {e}")
        else:
            failed.append(protein_id)
            print(f"✗ Missing protein file: {protein_id}")
    
    # Save statistics
    stats_file = "./experiments/results/phase2b/week1/physics_features_statistics.csv"
    with open(stats_file, 'w') as f:
        f.write("protein_id,num_nodes,num_edges,node_feat_dim,edge_physics_dim,pocket_residues,total_residues\n")
        for stats in statistics:
            f.write(f"{stats['protein_id']},{stats['num_nodes']},{stats['num_edges']},"
                   f"{stats['node_feat_dim']},{stats['edge_physics_dim']},"
                   f"{stats['pocket_residues']},{stats['total_residues']}\n")
    
    # Save processed IDs
    with open("./experiments/results/phase2b/week1/processed_physics_ids.txt", 'w') as f:
        for pid in processed:
            f.write(f"{pid}\n")
    
    print(f"\n✓ Physics graph processing complete!")
    print(f"  Successfully processed: {len(processed)}/{len(protein_ids)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Success rate: {len(processed)/len(protein_ids)*100:.1f}%")
    print(f"\n✓ Statistics saved to: {stats_file}")
    print(f"✓ Processed IDs saved to: ./experiments/results/phase2b/week1/processed_physics_ids.txt")
    
    if len(processed) > 0:
        # Save a sample graph for inspection
        sample_id = processed[0]
        sample_path = f"./data/processed/physics_graphs/train/{sample_id}_physics.pt"
        torch.save(torch.load(sample_path), 
                  "./experiments/results/phase2b/week1/physics_graph_sample.pt")
        print(f"✓ Sample graph saved to: ./experiments/results/phase2b/week1/physics_graph_sample.pt")
    
    return processed, failed

if __name__ == "__main__":
    process_physics_graphs()
