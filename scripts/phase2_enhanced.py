"""
PHASE 2 ENHANCED: Better features + Better model + Better training
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json

# Add src to path
sys.path.append('./src')

from models.graph_builder_fixed import GraphBuilderFixed
from models.enhanced_features import EnhancedFeatureBuilder
from models.improved_gnn import ImprovedGNN, FocalLoss
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class EnhancedGraphBuilder(GraphBuilderFixed):
    """Enhanced graph builder with better features"""
    def __init__(self, cutoff_distance=8.0, pocket_cutoff=4.0):
        super().__init__(cutoff_distance, pocket_cutoff)
        self.feature_builder = EnhancedFeatureBuilder()
    
    def create_enhanced_graph(self, pdb_file, ligand_file, protein_id="test"):
        """Create graph with enhanced features"""
        # Parse residues using parent method
        residues = self.parse_pdb(pdb_file)
        
        if not residues:
            print(f"No residues found in {pdb_file}")
            return None
        
        # Parse ligand coordinates
        ligand_coords = self.parse_ligand_coords(ligand_file)
        
        if len(ligand_coords) == 0:
            print(f"No ligand coordinates found in {ligand_file}")
            return None
        
        print(f"  Processing {protein_id}: {len(residues)} residues, {len(ligand_coords)} ligand atoms")
        
        # Create enhanced node features
        node_features = []
        positions = []
        
        for i, res in enumerate(residues):
            # Basic residue features
            residue_features = self.feature_builder.get_residue_features(res['name'])
            
            # Positional features
            positional_features = self.feature_builder.get_positional_features(i, len(residues))
            
            # Structural features
            centroid = res['centroid']
            structural_features = [
                centroid[0], centroid[1], centroid[2],  # Position
                res['radius_gyration'],  # Compactness
                len(res['atoms']) / 20.0  # Normalized atom count
            ]
            
            # Combine all features
            features = residue_features + positional_features + structural_features
            node_features.append(features)
            positions.append(centroid)
        
        # Convert to tensors
        node_features_array = np.array(node_features, dtype=np.float32)
        x = torch.from_numpy(node_features_array)
        
        positions_array = np.array(positions, dtype=np.float32)
        pos = torch.from_numpy(positions_array)
        
        # Create edges
        edges = self.create_edges_from_positions(positions_array)
        
        # Create pocket labels
        y_np = self.create_pocket_labels(residues, ligand_coords)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        pocket_count = int(y.sum().item())
        pocket_percent = pocket_count / len(residues) * 100
        print(f"  Pocket residues: {pocket_count}/{len(residues)} ({pocket_percent:.1f}%)")
        
        # Create graph
        graph = Data(
            x=x,
            edge_index=edges,
            y=y,
            pos=pos,
            protein_id=protein_id,
            num_nodes=len(residues),
            num_features=x.shape[1]
        )
        
        return graph
    
    def create_edges_from_positions(self, positions):
        """Create edges from positions (optimized)"""
        n = len(positions)
        edges = []
        
        # Use vectorized distance calculation for speed
        for i in range(n):
            # Calculate distances to all other points
            dists = np.linalg.norm(positions[i:i+1] - positions, axis=1)
            
            # Find indices within cutoff
            within_cutoff = np.where((dists < self.cutoff) & (dists > 0))[0]
            
            for j in within_cutoff:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index

def process_enhanced_proteins():
    """Process proteins with enhanced features"""
    print("\n" + "="*60)
    print("PROCESSING WITH ENHANCED FEATURES")
    print("="*60)
    
    # Read training IDs
    train_ids_path = './experiments/results/phase1/splits/train_ids.txt'
    with open(train_ids_path, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(train_ids)} training protein IDs")
    
    # Initialize enhanced builder
    builder = EnhancedGraphBuilder(cutoff_distance=8.0, pocket_cutoff=4.0)
    
    # Create output directory
    output_dir = './data/processed/graphs_enhanced/train'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process proteins (focus on first 100 for now)
    proteins_to_process = train_ids[:100]
    processed = []
    statistics = []
    
    for protein_id in tqdm(proteins_to_process, desc="Processing proteins"):
        protein_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.mol2"
        
        if not os.path.exists(ligand_path):
            ligand_path = f"./data/PDBbind/refined-set/{protein_id}/{protein_id}_ligand.sdf"
        
        if os.path.exists(protein_path) and os.path.exists(ligand_path):
            try:
                graph = builder.create_enhanced_graph(protein_path, ligand_path, protein_id)
                
                if graph is not None and graph.y.sum().item() > 0:
                    # Save graph
                    output_path = os.path.join(output_dir, f"{protein_id}_graph.pt")
                    torch.save(graph, output_path)
                    processed.append(protein_id)
                    
                    # Collect statistics
                    stats = {
                        'protein': protein_id,
                        'nodes': graph.num_nodes,
                        'features': graph.num_features,
                        'edges': graph.edge_index.shape[1],
                        'pockets': graph.y.sum().item(),
                        'pocket_percent': graph.y.sum().item() / graph.num_nodes * 100
                    }
                    statistics.append(stats)
                    
            except Exception as e:
                print(f"\nError processing {protein_id}: {e}")
    
    # Save statistics
    stats_dir = './experiments/results/phase2_enhanced'
    os.makedirs(stats_dir, exist_ok=True)
    
    with open(os.path.join(stats_dir, 'enhanced_statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\nProcessing complete: {len(processed)} proteins processed")
    
    # Show feature dimension
    if processed:
        sample_path = os.path.join(output_dir, f"{processed[0]}_graph.pt")
        sample_graph = torch.load(sample_path, weights_only=False)
        print(f"Feature dimension: {sample_graph.num_features}")
    
    return processed

def train_enhanced_model():
    """Train with enhanced features and model"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED MODEL")
    print("="*60)
    
    # Load processed graphs
    graph_dir = './data/processed/graphs_enhanced/train'
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    
    if len(graph_files) < 30:
        print(f"Need at least 30 graphs, but only have {len(graph_files)}")
        return None
    
    print(f"Loading {len(graph_files)} graphs...")
    
    graphs = []
    for graph_file in tqdm(graph_files, desc="Loading graphs"):
        graph_path = os.path.join(graph_dir, graph_file)
        graph = torch.load(graph_path, weights_only=False)
        graphs.append(graph)
    
    # Split data
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    
    # Calculate class statistics
    all_train_labels = torch.cat([g.y for g in train_graphs])
    num_pos = all_train_labels.sum().item()
    num_neg = len(all_train_labels) - num_pos
    
    print(f"Class distribution:")
    print(f"  Non-pocket: {num_neg} ({num_neg/len(all_train_labels)*100:.1f}%)")
    print(f"  Pocket: {num_pos} ({num_pos/len(all_train_labels)*100:.1f}%)")
    
    # Create data loaders with batch size > 1 for efficiency
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # Get feature dimension
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create enhanced model
    model = ImprovedGNN(input_dim=input_dim, hidden_dim=128, dropout=0.3)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Focal loss for imbalanced data
    # Alpha should be ~ (1 - positive_class_fraction)
    pos_fraction = num_pos / len(all_train_labels)
    alpha = 0.75  # Adjust based on imbalance
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # Training loop
    num_epochs = 50
    best_f1 = 0
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [], 'val_precision': [], 'val_recall': []
    }
    
    print("\nStarting training...")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch.y.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_loss, val_f1, val_precision, val_recall = evaluate_model(
            model, val_loader, device, criterion
        )
        
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            # Save model
            model_dir = './experiments/results/phase2_enhanced/models'
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
            
            print(f"  ↳ New best! F1: {val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"F1: {val_f1:.4f} | "
              f"P: {val_precision:.4f} | "
              f"R: {val_recall:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    best_model_path = './experiments/results/phase2_enhanced/models/best_model.pt'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Save final model and history
    torch.save(model.state_dict(), './experiments/results/phase2_enhanced/models/final_model.pt')
    with open('./experiments/results/phase2_enhanced/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history, best_f1

def evaluate_model(model, data_loader, device, criterion):
    """Evaluate model on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch.y.unsqueeze(1))
            
            # Get predictions
            probs = torch.sigmoid(output)
            
            # Dynamic threshold based on validation (could optimize)
            threshold = 0.3  # Lower threshold for imbalanced data
            preds = (probs > threshold).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())
            
            total_loss += loss.item()
            batch_count += 1
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_preds)) < 2:
        f1 = 0.0
        precision = 0.0 if all_preds[0] == 0 else 1.0
        recall = 0.0 if all_labels.sum() > 0 else 1.0
    else:
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return avg_loss, f1, precision, recall

def main():
    """Main execution"""
    print("="*70)
    print("PHYGNN-CPDD: PHASE 2 ENHANCED")
    print("="*70)
    
    # Step 1: Process proteins with enhanced features
    processed_ids = process_enhanced_proteins()
    
    if len(processed_ids) < 30:
        print(f"\nInsufficient data: {len(processed_ids)} proteins processed")
        print("Need at least 30 for training.")
        return
    
    # Step 2: Train enhanced model
    history, best_f1 = train_enhanced_model()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Compare with baseline
    baseline_f1 = 0.077  # From previous run
    improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
    
    print(f"Improvement over baseline: {improvement:+.1f}%")
    
    # Check target
    target_f1 = 0.30
    if best_f1 >= target_f1:
        print(f"\n🎯 TARGET ACHIEVED! F1 = {best_f1:.4f} >= {target_f1:.2f}")
    else:
        print(f"\n⚠️  Target not met: F1 = {best_f1:.4f} < {target_f1:.2f}")
        print("   Next steps:")
        print("   1. Process more proteins (100+)")
        print("   2. Add more advanced features")
        print("   3. Try ensemble methods")
        print("   4. Implement data augmentation")
    
    print(f"\nResults saved to ./experiments/results/phase2_enhanced/")

if __name__ == "__main__":
    main()
