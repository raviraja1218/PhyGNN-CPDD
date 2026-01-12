#!/usr/bin/env python3
"""
Fixed training for Phase 2C with adjusted hyperparameters
"""
import os
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
import time

sys.path.append('./src/models')
sys.path.append('./src/training')

# Import or define fixed model
class FixedHamiltonianGNN(torch.nn.Module):
    """Hamiltonian GNN with better regularization"""
    def __init__(self, input_dim=35, hidden_dim=128, lambda_physics=5e-5, dropout=0.3):
        super().__init__()
        self.lambda_physics = lambda_physics
        
        # Input projection with batch norm
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # GNN layers with residual connections
        self.conv1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        # Physics constraint parameters
        self.bond_weight = 1.0
        self.angle_weight = 0.5
    
    def forward(self, data):
        x = data.x
        
        # Input projection
        x = self.input_proj(x)
        x0 = x  # Residual
        
        # GNN layers with residuals
        x = torch.relu(self.conv1(x)) + x0
        x = torch.relu(self.conv2(x)) + x
        x = torch.relu(self.conv3(x)) + x
        
        # Predictions
        predictions = self.classifier(x)
        
        # Physics loss (simplified)
        physics_loss = self.calculate_physics_loss(data)
        
        return predictions, physics_loss
    
    def calculate_physics_loss(self, data):
        """Simplified physics loss"""
        pos = data.pos
        
        # Bond length loss (simplified - use distances between connected residues)
        if hasattr(data, 'edge_index') and data.edge_index.shape[1] > 0:
            edge_index = data.edge_index
            src, dst = edge_index[0], edge_index[1]
            bond_lengths = torch.norm(pos[src] - pos[dst], dim=1)
            
            # Ideal bond length ~3.8Å for residue-residue
            ideal_length = 3.8
            bond_loss = torch.mean((bond_lengths - ideal_length) ** 2)
        else:
            bond_loss = torch.tensor(0.0)
        
        # Total physics loss
        physics_loss = self.bond_weight * bond_loss
        
        return physics_loss

class FixedPhysicsTrainer:
    """Trainer with better hyperparameters"""
    def __init__(self, model, device='cuda', learning_rate=0.001, 
                 pos_weight=10.0, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Use more conservative pos_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'val_auc': [],
            'physics_loss': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_phys = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, physics_loss = self.model(batch)
            
            # Prediction loss
            pred_loss = self.criterion(predictions, batch.y.unsqueeze(1))
            
            # Combined loss with lower physics weight
            total_batch_loss = pred_loss + self.model.lambda_physics * physics_loss
            
            # Backward
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += pred_loss.item()
            total_phys += physics_loss.item()
        
        return total_loss / len(train_loader), total_phys / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                predictions, physics_loss = self.model(batch)
                
                # Loss
                pred_loss = self.criterion(predictions, batch.y.unsqueeze(1))
                total_loss += pred_loss.item()
                
                # Predictions
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())
        
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        if len(np.unique(all_preds)) == 1:
            f1 = 0.0
            precision = 0.0 if all_preds[0] == 0 else 1.0
            recall = 0.0 if np.sum(all_labels) > 0 else 1.0
        else:
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
        
        return total_loss/len(val_loader), f1, precision, recall, auc
    
    def train(self, train_loader, val_loader, epochs=100, 
              early_stopping_patience=25, save_dir=None):
        print("Starting fixed training...")
        print(f"Physics weight λ: {self.model.lambda_physics}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        best_f1 = 0
        patience = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_phys = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_f1, val_precision, val_recall, val_auc = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_f1)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_auc'].append(val_auc)
            self.history['physics_loss'].append(train_phys)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                if save_dir:
                    torch.save(self.model.state_dict(), f"{save_dir}/best_model.pt")
                print(f"  ↳ New best! F1: {val_f1:.4f}")
            else:
                patience += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            phys_ratio = train_phys / (train_loss + 1e-8)
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | "
                  f"P: {val_precision:.4f} | "
                  f"R: {val_recall:.4f} | "
                  f"Phys: {phys_ratio:.3f}")
            
            # Early stopping
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history

def train_fixed():
    """Run fixed training"""
    print("=" * 60)
    print("PHASE 2C FIXED TRAINING")
    print("=" * 60)
    
    # Load data
    from torch_geometric.loader import DataLoader
    import glob
    
    train_graphs = []
    val_graphs = []
    test_graphs = []
    
    # Load train graphs
    for f in glob.glob('./data/processed/phase2c_final_300/train/*.pt'):
        train_graphs.append(torch.load(f, weights_only=True))
    
    for f in glob.glob('./data/processed/phase2c_final_300/val/*.pt'):
        val_graphs.append(torch.load(f, weights_only=True))
    
    for f in glob.glob('./data/processed/phase2c_final_300/test/*.pt'):
        test_graphs.append(torch.load(f, weights_only=True))
    
    print(f"Loaded {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")
    
    # Check class distribution
    pos_count = 0
    total_count = 0
    for g in train_graphs:
        pos_count += g.y.sum().item()
        total_count += len(g.y)
    
    pos_ratio = pos_count / total_count
    print(f"Positive ratio: {pos_ratio:.3%}")
    print(f"Suggested pos_weight: {1/pos_ratio:.1f}")
    
    # Use more conservative pos_weight
    pos_weight = 10.0  # Instead of 20.0
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    # Create model with lower physics weight
    input_dim = train_graphs[0].x.shape[1]
    model = FixedHamiltonianGNN(
        input_dim=input_dim,
        hidden_dim=128,
        lambda_physics=5e-5,  # HALF of previous value
        dropout=0.3
    )
    
    # Create trainer
    trainer = FixedPhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        pos_weight=pos_weight,
        weight_decay=1e-4
    )
    
    # Create save directory
    save_dir = "./experiments/results/phase2c_fixed"
    os.makedirs(save_dir, exist_ok=True)
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=25,
        save_dir=save_dir
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        'test_performance': {
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'auc': float(test_auc),
            'loss': float(test_loss)
        },
        'training_info': {
            'train_graphs': len(train_graphs),
            'val_graphs': len(val_graphs),
            'test_graphs': len(test_graphs),
            'positive_ratio': float(pos_ratio),
            'pos_weight_used': float(pos_weight),
            'lambda_physics': float(model.lambda_physics),
            'training_time_minutes': train_time / 60,
            'epochs_trained': len(history['train_loss']),
            'best_val_f1': float(max(history['val_f1']))
        },
        'comparison': {
            'phase1_baseline': 0.077,
            'phase2a_base_gnn': 0.3077,
            'phase2b_70proteins': 0.5444,
            'phase2c_initial_300': 0.4066,
            'phase2c_fixed_300': float(test_f1)
        }
    }
    
    with open(f"{save_dir}/fixed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save history
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/final_model.pt")
    
    print("\n" + "=" * 60)
    print("FIXED TRAINING COMPLETE")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Training time: {train_time/60:.1f} minutes")
    
    if test_f1 > 0.45:
        print("✅ IMPROVEMENT ACHIEVED!")
    elif test_f1 > 0.4066:
        print("✅ SMALL IMPROVEMENT")
    else:
        print("⚠️ Need further tuning")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    train_fixed()
