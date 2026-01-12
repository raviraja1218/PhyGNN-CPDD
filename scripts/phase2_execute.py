"""
Phase 2 Execution Script
Process proteins and train Base GNN
"""
import os
import sys
import torch
from tqdm import tqdm

# Add src to path
sys.path.append('./src')

from models.graph_builder import GraphBuilder, test_builder
from models.base_gnn import SimpleGNN, test_model
from training.simple_trainer import SimpleTrainer, test_trainer
from torch_geometric.loader import DataLoader

def setup_directories():
    """Create necessary directories"""
    dirs = [
        './data/processed/graphs/train',
        './data/processed/graphs/val', 
        './data/processed/graphs/test',
        './experiments/results/phase2/models',
        './experiments/results/phase2/logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directories created")

def process_training_proteins():
    """Process all training proteins"""
    print("\n" + "="*50)
    print("PROCESSING TRAINING PROTEINS")
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
    builder = GraphBuilder(cutoff_distance=8.0)
    
    # Process proteins
    processed = []
    failed = []
    
    for protein_id in tqdm(train_ids[:50], desc="Processing first 50 proteins"):  # Start with 50
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2"
        
        if not os.path.exists(ligand_path):
            ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf"
        
        if os.path.exists(protein_path):
            try:
                graph = builder.create_graph(protein_path, ligand_path, protein_id)
                if graph is not None:
                    # Save graph
                    output_path = f"./data/processed/graphs/train/{protein_id}_graph.pt"
                    torch.save(graph, output_path)
                    processed.append(protein_id)
                else:
                    failed.append(protein_id)
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                failed.append(protein_id)
        else:
            failed.append(protein_id)
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {len(processed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Success rate: {len(processed)/(len(processed)+len(failed))*100:.1f}%")
    
    # Save processed IDs
    with open('./experiments/results/phase2/processed_ids.txt', 'w') as f:
        for pid in processed:
            f.write(f"{pid}\n")
    
    return processed

def load_processed_graphs(protein_ids):
    """Load processed graphs"""
    graphs = []
    for pid in protein_ids:
        graph_path = f"./data/processed/graphs/train/{pid}_graph.pt"
        if os.path.exists(graph_path):
            try:
                graph = torch.load(graph_path, weights_only=False)
                graphs.append(graph)
            except Exception as e:
                print(f"Error loading {pid}: {e}")
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs

def train_base_gnn(graphs):
    """Train Base GNN"""
    print("\n" + "="*50)
    print("TRAINING BASE GNN")
    print("="*50)
    
    if len(graphs) < 10:
        print(f"Need at least 10 graphs, but only have {len(graphs)}")
        return None
    
    # Split into train/validation
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    
    # Get input dimension from first graph
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = SimpleGNN(input_dim=input_dim, hidden_dim=64)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = SimpleTrainer(model, device=device, lr=0.001)
    
    # Train
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=20,
        save_dir='./experiments/results/phase2/models'
    )
    
    return history

def main():
    """Main execution function"""
    print("="*60)
    print("PHYGNN-CPDD: PHASE 2 EXECUTION")
    print("="*60)
    
    # Step 1: Setup
    setup_directories()
    
    # Step 2: Test components
    print("\nTesting components...")
    test_builder()
    test_model()
    test_trainer()
    
    # Step 3: Process proteins
    processed_ids = process_training_proteins()
    
    if not processed_ids:
        print("No proteins processed. Exiting.")
        return
    
    # Step 4: Load graphs
    graphs = load_processed_graphs(processed_ids)
    
    if not graphs:
        print("No graphs loaded. Exiting.")
        return
    
    # Step 5: Train Base GNN
    history = train_base_gnn(graphs)
    
    if history:
        print("\n" + "="*50)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best validation F1: {max(history['val_f1']):.4f}")
        print(f"Final results saved to ./experiments/results/phase2/")
    else:
        print("\nPhase 2 training failed.")

if __name__ == "__main__":
    main()
