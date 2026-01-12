#!/usr/bin/env python3
"""
PHASE 2C FINAL FIXED TRAINING
Using CORRECTLY labeled data from Phase 2B builder
"""
import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
import time

sys.path.append('./src/models')
sys.path.append('./src/training')

# Import Phase 2B PROVEN model
try:
    from improved_gnn_fixed import ImprovedGNN
    ModelClass = ImprovedGNN
    print("Using ImprovedGNN (Phase 2B model)")
except:
    # Create simple model matching Phase 2B
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, BatchNorm
    
    class SimpleGNNPhase2B(nn.Module):
        """Simple GNN matching Phase 2B architecture"""
        def __init__(self, input_dim=30, hidden_dim=128, output_dim=1, dropout=0.3):
            super().__init__()
            self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
            self.gat2 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.gat3 = GATConv(hidden_dim*4, hidden_dim, dropout=dropout)
            
            self.bn1 = BatchNorm(hidden_dim*4)
            self.bn2 = BatchNorm(hidden_dim*4)
            self.bn3 = BatchNorm(hidden_dim)
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, hidden_dim//4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//4, output_dim)
            )
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            
            x = F.elu(self.gat1(x, edge_index))
            x = self.bn1(x)
            x = F.dropout(x, p=0.3, training=self.training)
            
            x = F.elu(self.gat2(x, edge_index))
            x = self.bn2(x)
            x = F.dropout(x, p=0.3, training=self.training)
            
            x = self.gat3(x, edge_index)
            x = self.bn3(x)
            
            return self.classifier(x)
    
    ModelClass = SimpleGNNPhase2B
    print("Using SimpleGNNPhase2B (Phase 2B architecture)")

from physics_trainer import PhysicsTrainer

def load_correctly_labeled_graphs():
    """Load graphs with CORRECT labels"""
    print("Loading correctly labeled graphs...")
    
    # Try corrected labels first
    graph_dir = './data/processed/phase2c_correct_labels/'
    if not os.path.exists(graph_dir):
        print(f"ERROR: Corrected graphs not found at {graph_dir}")
        print("Run phase2c_fix_labels.py first!")
        return None, None, None
    
    files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    print(f"Found {len(files)} correctly labeled graphs")
    
    graphs = []
    for f in tqdm(files, desc="Loading"):
        try:
            graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
            # Verify has positive labels
            if graph.y.sum().item() > 0:
                graphs.append(graph)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    print(f"Loaded {len(graphs)} graphs WITH POSITIVE LABELS")
    
    if len(graphs) < 10:
        print("ERROR: Not enough graphs with positive labels")
        return None, None, None
    
    # Split
    train_size = int(0.7 * len(graphs))
    val_size = int(0.15 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]
    
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs

def final_training_with_correct_labels():
    """FINAL training with CORRECT labels"""
    print("=" * 70)
    print("PHASE 2C FINAL: TRAINING WITH CORRECT LABELS")
    print("=" * 70)
    
    # Load correctly labeled graphs
    train_graphs, val_graphs, test_graphs = load_correctly_labeled_graphs()
    if train_graphs is None:
        return None
    
    # Check class balance
    total_pos = sum(g.y.sum().item() for g in train_graphs)
    total_nodes = sum(g.y.shape[0] for g in train_graphs)
    pos_ratio = total_pos / total_nodes if total_nodes > 0 else 0.046
    
    print(f"Class balance: {pos_ratio:.3%} positive ({total_pos}/{total_nodes})")
    
    # PHASE 2B HYPERPARAMETERS (PROVEN)
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 8,
        'hidden_dim': 128,
        'pos_weight': 20.0,  # From Phase 2B
        'dropout': 0.3,
        'epochs': 100,
        'patience': 20
    }
    
    print("\nPHASE 2B PROVEN HYPERPARAMETERS:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create model
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    model = ModelClass(
        input_dim=input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        dropout=hyperparams['dropout']
    )
    
    # Simple trainer (NO PHYSICS for now - get basics working)
    from torch_geometric.loader import DataLoader
    from torch.optim import Adam
    import torch.nn as nn
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyperparams['pos_weight']]).to(device))
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING FINAL TRAINING")
    print("=" * 70)
    
    best_f1 = 0
    best_model_state = None
    train_losses = []
    val_f1s = []
    
    start_time = time.time()
    
    for epoch in range(hyperparams['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch)
            loss = criterion(logits, batch.y.unsqueeze(1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())
        
        # Calculate F1
        from sklearn.metrics import f1_score
        if len(np.unique(all_preds)) > 1 and len(np.unique(all_labels)) > 1:
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        else:
            val_f1 = 0.0
        
        val_f1s.append(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{hyperparams['epochs']} | "
                  f"Train Loss: {train_losses[-1]:.4f} | "
                  f"Val F1: {val_f1:.4f}")
        
        # Early stopping
        if len(val_f1s) > hyperparams['patience']:
            if max(val_f1s[-hyperparams['patience']:]) < best_f1:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test
    print("\nEvaluating on test set...")
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            test_preds.extend(preds.cpu().numpy().flatten())
            test_labels.extend(batch.y.cpu().numpy().flatten())
            test_probs.extend(probs.cpu().numpy().flatten())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    if len(np.unique(test_preds)) > 1 and len(np.unique(test_labels)) > 1:
        test_f1 = f1_score(test_labels, test_preds, zero_division=0)
        test_precision = precision_score(test_labels, test_preds, zero_division=0)
        test_recall = recall_score(test_labels, test_preds, zero_division=0)
    else:
        test_f1 = test_precision = test_recall = 0.0
    
    # AUC
    if len(np.unique(test_labels)) > 1:
        test_auc = roc_auc_score(test_labels, test_probs)
    else:
        test_auc = 0.5
    
    # RESULTS
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'best_val_f1': float(best_f1),
        'phase2b_baseline': 0.5444,
        'improvement_over_phase2b': float(test_f1 - 0.5444),
        'training_time_minutes': float(training_time / 60),
        'num_train_graphs': len(train_graphs),
        'num_val_graphs': len(val_graphs),
        'num_test_graphs': len(test_graphs),
        'class_balance': float(pos_ratio),
        'achieved_target': bool(test_f1 > 0.60)
    }
    
    # Save
    output_dir = './experiments/results/phase2c/final_correct_labels'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), f'{output_dir}/model.pt')
    
    # PRINT
    print("\n" + "=" * 70)
    print("FINAL RESULTS WITH CORRECT LABELS")
    print("=" * 70)
    print(f"Test F1:       {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Best Val F1:    {best_f1:.4f}")
    print(f"Training time:  {training_time/60:.1f} min")
    print(f"Target (>0.60): {'✅ ACHIEVED' if test_f1 > 0.60 else '❌ NOT ACHIEVED'}")
    print("=" * 70)
    
    if test_f1 > 0.60:
        print("\n🎉 PHASE 2C TARGET ACHIEVED WITH CORRECT LABELS!")
        
        # Update success flag
        with open('./experiments/results/phase2c/PHASE2C_SUCCESS_REAL.txt', 'w') as f:
            f.write(f"PHASE 2C COMPLETE - REAL SUCCESS\n")
            f.write(f"Test F1: {test_f1:.4f}\n")
            f.write(f"Target: >0.60\n")
            f.write(f"Achieved: YES\n")
            f.write(f"Method: Correct labels + Phase 2B recipe\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        print(f"\n⚠️ Below target but better than 0.0000")
        print(f"   Phase 2B still gives us 0.5444")
    
    return results

if __name__ == "__main__":
    results = final_training_with_correct_labels()
