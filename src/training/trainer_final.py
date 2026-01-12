"""
Physics trainer for Hamiltonian GNN
Based on Phase 2B working code
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

class PhysicsTrainer:
    def __init__(self, model, device='cuda', learning_rate=0.001, 
                 weight_decay=1e-5, pos_weight=None, lambda_physics=0.0001):
        """
        Initialize trainer with physics constraints
        
        Args:
            model: HamiltonianGNN model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            pos_weight: Weight for positive class
            lambda_physics: Weight for physics loss (from Phase 2B discovery)
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_physics = lambda_physics
        
        # Prediction loss
        if pos_weight is not None:
            self.prediction_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(device)
            )
        else:
            self.prediction_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
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
            'prediction_loss': [],
            'physics_loss': [],
            'physics_loss_ratio': [],
            'learning_rate': []
        }
        
        self.best_f1 = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_pred_loss = 0
        total_physics_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with physics loss
            logits, physics_loss = self.model(batch)
            
            # Calculate prediction loss
            pred_loss = self.prediction_criterion(logits, batch.y.unsqueeze(1))
            
            # Combine losses
            loss = pred_loss + self.lambda_physics * physics_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            total_loss += loss.item() * batch_size
            total_pred_loss += pred_loss.item() * batch_size
            total_physics_loss += physics_loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'pred': pred_loss.item(),
                'phys': physics_loss.item()
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_pred_loss = total_pred_loss / total_samples if total_samples > 0 else 0
        avg_physics_loss = total_physics_loss / total_samples if total_samples > 0 else 0
        physics_ratio = avg_physics_loss / (avg_physics_loss + avg_pred_loss) if (avg_physics_loss + avg_pred_loss) > 0 else 0
        
        return avg_loss, avg_pred_loss, avg_physics_loss, physics_ratio
    
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
                logits, physics_loss = self.model(batch)
                
                # Calculate prediction loss
                pred_loss = self.prediction_criterion(logits, batch.y.unsqueeze(1))
                loss = pred_loss + self.lambda_physics * physics_loss
                
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
        
        # Handle edge cases
        if len(np.unique(all_preds)) == 1:
            f1 = 0.0
            precision = 0.0 if all_preds[0] == 0 else 1.0
            recall = 0.0 if all_labels.sum() > 0 else 1.0
        else:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # AUC
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        return avg_loss, f1, precision, recall, auc
    
    def train(self, train_loader, val_loader, epochs=100, 
              early_stopping_patience=20, save_dir=None, verbose=True):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save models
            verbose: Print progress
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        if verbose:
            print(f"Starting training for {epochs} epochs")
            print(f"Training on {len(train_loader.dataset)} graphs")
            print(f"Validating on {len(val_loader.dataset)} graphs")
            print(f"Physics weight λ: {self.lambda_physics}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, pred_loss, physics_loss, physics_ratio = self.train_epoch(train_loader)
            
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
            self.history['prediction_loss'].append(pred_loss)
            self.history['physics_loss'].append(physics_loss)
            self.history['physics_loss_ratio'].append(physics_ratio)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                if save_dir:
                    torch.save(self.model.state_dict(), 
                              os.path.join(save_dir, 'best_model.pt'))
                patience_counter = 0
                if verbose:
                    print(f"  ↳ New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"LR: {self.history['learning_rate'][-1]:.6f} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"F1: {val_f1:.4f} | "
                      f"Phys Ratio: {physics_ratio:.3f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model
        if save_dir:
            torch.save(self.model.state_dict(), 
                      os.path.join(save_dir, 'final_model.pt'))
            
            # Save training history
            with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best validation F1: {self.best_f1:.4f}")
        
        return self.history

# Test function
def test_physics_trainer():
    """Test the physics trainer"""
    print("Testing PhysicsTrainer...")
    
    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, data):
            logits = self.linear(data.x)
            physics_loss = torch.tensor(0.1)  # Dummy physics loss
            return logits, physics_loss
    
    # Create dummy data
    from torch_geometric.data import Data, Batch
    
    graphs = []
    for i in range(5):
        graph = Data(
            x=torch.randn(20, 10),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.randint(0, 2, (20,)).float()
        )
        graphs.append(graph)
    
    train_loader = DataLoader(graphs[:3], batch_size=1)
    val_loader = DataLoader(graphs[3:], batch_size=1)
    
    # Create model and trainer
    model = SimpleModel()
    trainer = PhysicsTrainer(model, device='cpu', lambda_physics=0.0001)
    
    # Train for 2 epochs
    history = trainer.train(train_loader, val_loader, epochs=2, verbose=False)
    
    print(f"Training history keys: {list(history.keys())}")
    print(f"Best F1: {trainer.best_f1:.4f}")
    print("\nPhysicsTrainer test PASSED!")
    return True

if __name__ == "__main__":
    test_physics_trainer()
