"""
Phase 2 Execution with Fixed Graph Builder
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append('./src')

try:
    from models.graph_builder_fixed import GraphBuilderFixed
except ImportError:
    print("Creating GraphBuilderFixed module...")
    # Create the module inline if needed
    import types
    GraphBuilderFixed = types.SimpleNamespace()

from models.base_gnn import SimpleGNN
from training.simple_trainer import SimpleTrainer
from torch_geometric.loader import DataLoader

def setup_directories():
    """Create necessary directories"""
    dirs = [
        './data/processed/graphs_fixed/train',
        './data/processed/graphs_fixed/val', 
        './data/processed/graphs_fixed/test',
        './experiments/results/phase2_fixed/models',
        './experiments/results/phase2_fixed/logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directories created")

def process_proteins_with_labels():
    """Process proteins with proper pocket labels"""
    print("\n" + "="*50)
    print("PROCESSING PROTEINS WITH PROPER POCKET LABELS")
    print("="*50)
    
    # Read training IDs from Phase 1
    train_ids_path = './experiments/results/phase1/splits/train_ids.txt'
    if not os.path.exists(train_ids_path):
        print(f"Error: {train_ids_path} not found")
        return []
    
    with open(train_ids_path, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(train_ids)} training protein IDs")
    
    # Initialize graph builder
    builder = GraphBuilderFixed(cutoff_distance=8.0, pocket_cutoff=4.0)
    
    # Process first 50 proteins
    proteins_to_process = train_ids[:50]
    processed = []
    failed = []
    pocket_stats = []
    
    for protein_id in tqdm(proteins_to_process, desc="Processing proteins"):
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2"
        
        if not os.path.exists(ligand_path):
            ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf"
        
        if os.path.exists(protein_path) and os.path.exists(ligand_path):
            try:
                print(f"\nProcessing {protein_id}...")
                graph = builder.create_graph(protein_path, ligand_path, protein_id)
                
                if graph is not None:
                    # Check if graph has pocket labels
                    pocket_count = graph.y.sum().item()
                    if pocket_count > 0:  # Only keep proteins with pockets
                        # Save graph
                        output_path = f"./data/processed/graphs_fixed/train/{protein_id}_graph.pt"
                        torch.save(graph, output_path)
                        processed.append(protein_id)
                        
                        # Collect statistics
                        pocket_percent = pocket_count / graph.num_nodes * 100
                        pocket_stats.append({
                            'protein': protein_id,
                            'nodes': graph.num_nodes,
                            'pockets': pocket_count,
                            'percent': pocket_percent
                        })
                        
                        print(f"  ✓ Saved: {graph.num_nodes} nodes, {pocket_count} pockets ({pocket_percent:.1f}%)")
                    else:
                        print(f"  ✗ No pockets found in {protein_id}")
                        failed.append(protein_id)
                else:
                    failed.append(protein_id)
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                failed.append(protein_id)
        else:
            failed.append(protein_id)
            print(f"Missing files for {protein_id}")
    
    # Print statistics
    print(f"\n" + "="*50)
    print("PROCESSING STATISTICS")
    print("="*50)
    print(f"Successfully processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    
    if processed:
        # Calculate average pocket percentage
        avg_pocket = np.mean([s['percent'] for s in pocket_stats])
        print(f"Average pocket residues: {avg_pocket:.1f}%")
        
        # Show top 5 proteins with most pockets
        print("\nTop 5 proteins by pocket percentage:")
        sorted_stats = sorted(pocket_stats, key=lambda x: x['percent'], reverse=True)
        for i, stat in enumerate(sorted_stats[:5]):
            print(f"  {i+1}. {stat['protein']}: {stat['pockets']}/{stat['nodes']} ({stat['percent']:.1f}%)")
    
    # Save processed IDs
    with open('./experiments/results/phase2_fixed/processed_ids.txt', 'w') as f:
        for pid in processed:
            f.write(f"{pid}\n")
    
    # Save statistics
    import json
    with open('./experiments/results/phase2_fixed/pocket_statistics.json', 'w') as f:
        json.dump(pocket_stats, f, indent=2)
    
    return processed

def load_graphs(protein_ids):
    """Load processed graphs"""
    graphs = []
    for pid in protein_ids:
        graph_path = f"./data/processed/graphs_fixed/train/{pid}_graph.pt"
        if os.path.exists(graph_path):
            try:
                graph = torch.load(graph_path, weights_only=False)
                graphs.append(graph)
            except Exception as e:
                print(f"Error loading {pid}: {e}")
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Calculate overall statistics
    if graphs:
        total_nodes = sum(g.num_nodes for g in graphs)
        total_pockets = sum(g.y.sum().item() for g in graphs)
        print(f"Total nodes: {total_nodes}")
        print(f"Total pocket residues: {total_pockets}")
        print(f"Overall pocket percentage: {total_pockets/total_nodes*100:.2f}%")
    
    return graphs

def train_model(graphs):
    """Train GNN model"""
    print("\n" + "="*50)
    print("TRAINING GNN WITH PROPER LABELS")
    print("="*50)
    
    if len(graphs) < 10:
        print(f"Need at least 10 graphs, but only have {len(graphs)}")
        return None
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    
    # Calculate class weights (for imbalanced data)
    all_train_labels = torch.cat([g.y for g in train_graphs])
    num_pos = all_train_labels.sum().item()
    num_neg = len(all_train_labels) - num_pos
    
    print(f"Class distribution in training set:")
    print(f"  Non-pocket (0): {num_neg} ({num_neg/len(all_train_labels)*100:.1f}%)")
    print(f"  Pocket (1): {num_pos} ({num_pos/len(all_train_labels)*100:.1f}%)")
    
    # Class weight for loss function (inverse frequency)
    pos_weight = num_neg / max(num_pos, 1)
    print(f"Positive class weight for loss: {pos_weight:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    
    # Get input dimension from first graph
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = SimpleGNN(input_dim=input_dim, hidden_dim=64)
    
    # Create trainer with class weighting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # We'll modify the trainer to use weighted loss
    from torch import nn
    
    class WeightedTrainer(SimpleTrainer):
        def __init__(self, model, device='cuda', lr=0.001, pos_weight=1.0):
            super().__init__(model, device, lr)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    trainer = WeightedTrainer(model, device=device, lr=0.001, pos_weight=pos_weight)
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=30,  # More epochs for learning
        save_dir='./experiments/results/phase2_fixed/models'
    )
    
    return history

def main():
    """Main execution function"""
    print("="*60)
    print("PHYGNN-CPDD: PHASE 2 WITH FIXED LABELS")
    print("="*60)
    
    # Step 1: Setup
    setup_directories()
    
    # Step 2: Test fixed builder
    print("\nTesting fixed graph builder...")
    try:
        from models.graph_builder_fixed import test_fixed_builder
        test_fixed_builder()
    except:
        print("Could not run test, but will proceed anyway")
    
    # Step 3: Process proteins with proper labels
    processed_ids = process_proteins_with_labels()
    
    if len(processed_ids) < 20:
        print(f"\nWARNING: Only {len(processed_ids)} proteins processed.")
        print("Need at least 20 for meaningful training.")
        if len(processed_ids) < 10:
            print("Exiting due to insufficient data.")
            return
    
    # Step 4: Load graphs
    graphs = load_graphs(processed_ids)
    
    if len(graphs) < 10:
        print("Not enough graphs loaded. Exiting.")
        return
    
    # Step 5: Train model
    history = train_model(graphs)
    
    if history:
        print("\n" + "="*50)
        print("PHASE 2 WITH FIXED LABELS COMPLETED!")
        print("="*50)
        best_f1 = max(history['val_f1'])
        print(f"Best validation F1: {best_f1:.4f}")
        
        # Check if we met the target
        target_f1 = 0.30
        if best_f1 >= target_f1:
            print(f"✓ TARGET ACHIEVED: F1 > {target_f1:.2f}")
        else:
            print(f"✗ TARGET NOT MET: F1 = {best_f1:.4f} (target: {target_f1:.2f})")
            print("  Possible issues:")
            print("  1. Still insufficient pocket labels")
            print("  2. Model architecture too simple")
            print("  3. Need more training data")
            print("  4. Need better features")
        
        print(f"\nFinal results saved to ./experiments/results/phase2_fixed/")
    else:
        print("\nTraining failed.")

if __name__ == "__main__":
    main()
