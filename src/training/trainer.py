"""
Training pipeline for Base GNN
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import time
import os
from tqdm import tqdm
import json

class GNNtrainer:
    def __init__(self, model, device='cuda', learning_rate=0.001, 
                 weight_decay=1e-5, pos_weight=None):
        """
        Initialize trainer
        
        Args:
            model: GNN model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            pos_weight: Weight for positive class (pocket residues are rare)
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function with class weighting (pocket residues are rare)
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=learning_rate, 
                                         weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        self.best_f1 = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            
            # Calculate loss
            loss = self.criterion(logits, batch.y.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / total_samples if total_samples > 0 else 0
    
    def evaluate(self, val_loader, threshold=0.5):
        """Evaluate model on validation set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                
                # Calculate loss
                loss = self.criterion(logits, batch.y.unsqueeze(1))
                
                # Get predictions
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                # Collect results
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())
                
                # Update loss
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Handle edge cases (no positive predictions)
        if len(np.unique(all_preds)) == 1:
            f1 = 0.0
            precision = 0.0 if all_preds[0] == 0 else 1.0
            recall = 0.0 if all_labels.sum() > 0 else 1.0
        else:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # AUC (handle case with only one class)
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        return avg_loss, f1, precision, recall, auc
    
    def train(self, train_loader, val_loader, epochs=100, 
              early_stopping_patience=20, save_dir='./experiments/results/phase2/models'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs")
        print(f"Training on {len(train_loader.dataset)} graphs")
        print(f"Validating on {len(val_loader.dataset)} graphs")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_f1, val_precision, val_recall, val_auc = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_f1)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'base_gnn_best.pt'))
                patience_counter = 0
                print(f"  ↳ New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {self.history['learning_rate'][-1]:.6f} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | "
                  f"P: {val_precision:.4f} | "
                  f"R: {val_recall:.4f} | "
                  f"AUC: {val_auc:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(save_dir, 'base_gnn_final.pt'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation F1: {self.best_f1:.4f}")
        
        return self.history

# Test function
def test_trainer():
    """Test the training pipeline"""
    from src.models.base_gnn import BaseGNN
    from src.models.graph_builder import GraphBuilder
    
    print("Testing training pipeline...")
    
    # Create a small dummy dataset
    builder = GraphBuilder()
    sample_dir = "./data/PDBbind/refined-set"
    
    # Find first 3 proteins
    proteins = []
    count = 0
    for folder in os.listdir(sample_dir):
        if count >= 3:
            break
        protein_path = os.path.join(sample_dir, folder, f"{folder}_protein.pdb")
        ligand_path = os.path.join(sample_dir, folder, f"{folder}_ligand.mol2")
        
        if os.path.exists(protein_path):
            graph = builder.build_from_pdb(protein_path, ligand_path, folder)
            if graph is not None:
                proteins.append(graph)
                count += 1
    
    if len(proteins) < 2:
        print("Need at least 2 proteins for testing")
        return False
    
    print(f"Created dataset with {len(proteins)} proteins")
    
    # Create data loaders
    train_loader = DataLoader(proteins[:2], batch_size=1, shuffle=True)
    val_loader = DataLoader(proteins[2:], batch_size=1, shuffle=False)
    
    # Create model
    model = BaseGNN(input_dim=proteins[0].x.shape[1], hidden_dim=32, num_layers=2)
    
    # Create trainer (use CPU for testing)
    trainer = GNNtrainer(model, device='cpu', learning_rate=0.001)
    
    # Train for 2 epochs (just to test)
    history = trainer.train(train_loader, val_loader, epochs=2, 
                           save_dir='./experiments/results/phase2/test_models')
    
    print("\nTrainer test PASSED!")
    return True

if __name__ == "__main__":
    test_trainer()
