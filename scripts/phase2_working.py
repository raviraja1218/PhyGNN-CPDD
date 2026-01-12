"""
WORKING PHASE 2 PIPELINE - Simple but effective
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json

# Add src to path
sys.path.append('./src')

try:
    from models.simple_enhanced_builder import SimpleEnhancedBuilder
    from models.improved_gnn_fixed import ImprovedGNNFixed, FocalLoss
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating modules inline...")
    
    # Define simple versions if import fails
    class SimpleEnhancedBuilder:
        def __init__(self):
            pass
    
    class ImprovedGNNFixed:
        def __init__(self):
            pass
    
    class FocalLoss:
        def __init__(self):
            pass

from torch_geometric.loader import DataLoader

def setup_directories():
    """Setup directories"""
    dirs = [
        './data/processed/graphs_working/train',
        './data/processed/graphs_working/val',
        './experiments/results/phase2_working/models',
        './experiments/results/phase2_working/logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directories created")

def process_proteins_robust():
    """Process proteins robustly"""
    print("\n" + "="*60)
    print("PROCESSING PROTEINS (ROBUST VERSION)")
    print("="*60)
    
    # Read training IDs
    train_ids_path = './experiments/results/phase1/splits/train_ids.txt'
    if not os.path.exists(train_ids_path):
        print(f"Error: {train_ids_path} not found")
        return []
    
    with open(train_ids_path, 'r') as f:
        all_train_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(all_train_ids)} training protein IDs")
    
    # Use first 100 for speed
    train_ids = all_train_ids[:100]
    print(f"Processing first {len(train_ids)} proteins")
    
    # Initialize builder
    builder = SimpleEnhancedBuilder(cutoff_distance=8.0, pocket_cutoff=4.0)
    
    processed = []
    failed = []
    
    for protein_id in tqdm(train_ids, desc="Processing"):
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        
        # Try multiple ligand file formats
        ligand_formats = [
            f"{protein_id}_ligand.mol2",
            f"{protein_id}_ligand.sdf",
            f"{protein_id}_ligand.pdb"
        ]
        
        ligand_path = None
        for fmt in ligand_formats:
            test_path = f"./data/PDBbind/refined-set/{protein_id}/{fmt}"
            if os.path.exists(test_path):
                ligand_path = test_path
                break
        
        if not os.path.exists(protein_path):
            failed.append(protein_id)
            continue
        
        if ligand_path is None:
            # Try to find any ligand file in directory
            protein_dir = f"./data/PDBbind/refined-set/{protein_id}"
            if os.path.exists(protein_dir):
                for fname in os.listdir(protein_dir):
                    if 'ligand' in fname.lower() and fname.endswith(('.mol2', '.sdf', '.pdb')):
                        ligand_path = os.path.join(protein_dir, fname)
                        break
        
        if ligand_path is None:
            failed.append(protein_id)
            continue
        
        try:
            graph = builder.create_enhanced_graph(protein_path, ligand_path, protein_id)
            
            if graph is not None and graph.y.sum().item() > 0:
                # Save graph
                output_path = f"./data/processed/graphs_working/train/{protein_id}_graph.pt"
                torch.save(graph, output_path, pickle_protocol=4)
                processed.append(protein_id)
            else:
                failed.append(protein_id)
                
        except Exception as e:
            print(f"\nError with {protein_id}: {e}")
            failed.append(protein_id)
    
    # Report
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {len(processed)}")
    print(f"  Failed: {len(failed)}")
    
    if processed:
        # Save processed IDs
        with open('./experiments/results/phase2_working/processed_ids.txt', 'w') as f:
            for pid in processed:
                f.write(f"{pid}\n")
        
        # Load first graph to check features
        sample_path = f"./data/processed/graphs_working/train/{processed[0]}_graph.pt"
        sample_graph = torch.load(sample_path, weights_only=False)
        print(f"\nSample graph features: {sample_graph.num_features} dimensions")
        print(f"Sample graph nodes: {sample_graph.num_nodes}")
        print(f"Sample graph pockets: {sample_graph.y.sum().item()}")
    
    return processed

def load_graphs_safely(protein_ids):
    """Load graphs safely"""
    graphs = []
    
    for pid in tqdm(protein_ids, desc="Loading graphs"):
        graph_path = f"./data/processed/graphs_working/train/{pid}_graph.pt"
        
        if os.path.exists(graph_path):
            try:
                graph = torch.load(graph_path, weights_only=False)
                
                # Basic validation
                if (graph.x.dim() == 2 and graph.edge_index.dim() == 2 and 
                    graph.y.dim() == 1 and graph.x.size(0) == graph.y.size(0)):
                    graphs.append(graph)
                else:
                    print(f"  Invalid graph structure for {pid}")
                    
            except Exception as e:
                print(f"  Error loading {pid}: {e}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs

def train_simple_model(graphs):
    """Train a simple but effective model"""
    print("\n" + "="*60)
    print("TRAINING SIMPLE MODEL")
    print("="*60)
    
    if len(graphs) < 20:
        print(f"Need at least 20 graphs, but only have {len(graphs)}")
        return None, 0
    
    # Split data
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    
    # Calculate class balance
    all_labels = torch.cat([g.y for g in train_graphs])
    pos_count = all_labels.sum().item()
    neg_count = len(all_labels) - pos_count
    
    print(f"Class balance: {neg_count} non-pocket, {pos_count} pocket")
    print(f"Positive ratio: {pos_count/len(all_labels)*100:.1f}%")
    
    # Get feature dimension
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create simplified model (not too complex)
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
            self.conv2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.conv3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, 1)
            
        def forward(self, data):
            x = data.x
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return self.classifier(x)
    
    model = SimpleModel(input_dim=input_dim, hidden_dim=64)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Weighted loss for imbalance
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    
    # Training
    num_epochs = 30
    best_f1 = 0
    history = {'train_loss': [], 'val_f1': []}
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            loss = criterion(output, batch.y.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / max(batch_count, 1)
        history['train_loss'].append(avg_train_loss)
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                probs = torch.sigmoid(output)
                preds = (probs > 0.3).float()  # Lower threshold for imbalance
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())
        
        # Calculate F1
        from sklearn.metrics import f1_score
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_preds)) < 2:
            val_f1 = 0.0
        else:
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), './experiments/results/phase2_working/models/best_model.pt')
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        # Early stopping if no improvement for 5 epochs
        if epoch > 5 and max(history['val_f1'][-5:]) <= best_f1:
            print("Early stopping")
            break
    
    # Save history
    with open('./experiments/results/phase2_working/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history, best_f1

def main():
    """Main function"""
    print("="*70)
    print("PHYGNN-CPDD: WORKING PHASE 2 PIPELINE")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Process proteins
    processed_ids = process_proteins_robust()
    
    if len(processed_ids) < 20:
        print(f"\nERROR: Only {len(processed_ids)} proteins processed.")
        print("Need at least 20 for training.")
        print("\nTroubleshooting:")
        print("1. Check if data directory exists: ./data/PDBbind/refined-set/")
        print("2. Check if files have correct permissions")
        print("3. Try processing manually:")
        print("   python3 ./src/models/simple_enhanced_builder.py")
        return
    
    # Load graphs
    graphs = load_graphs_safely(processed_ids)
    
    if len(graphs) < 20:
        print(f"Only {len(graphs)} graphs loaded. Need at least 20.")
        return
    
    # Train
    history, best_f1 = train_simple_model(graphs)
    
    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Compare with baseline and target
    baseline_f1 = 0.077
    target_f1 = 0.30
    
    if best_f1 > baseline_f1:
        improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
        print(f"Improvement over baseline: +{improvement:.1f}%")
    
    if best_f1 >= target_f1:
        print(f"\n🎉 TARGET ACHIEVED! F1 = {best_f1:.4f} >= {target_f1:.2f}")
    else:
        print(f"\nCurrent F1: {best_f1:.4f} (Target: {target_f1:.2f})")
        
        if best_f1 < 0.15:
            print("\n🚨 CRITICAL: F1 < 0.15")
            print("Possible issues:")
            print("1. Features not informative enough")
            print("2. Class imbalance too severe")
            print("3. Model too simple")
            print("\nImmediate fixes:")
            print("1. Add more features (secondary structure, conservation)")
            print("2. Use stronger data augmentation")
            print("3. Try different model architecture")
        elif best_f1 < 0.25:
            print("\n⚠️  MODERATE: F1 < 0.25")
            print("Getting closer! Try:")
            print("1. Process more proteins (200+)")
            print("2. Fine-tune hyperparameters")
            print("3. Add graph pooling layers")
    
    print(f"\nResults saved to ./experiments/results/phase2_working/")

if __name__ == "__main__":
    main()
