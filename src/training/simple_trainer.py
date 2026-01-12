"""
Simple Trainer for GNN
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import os

class SimpleTrainer:
    def __init__(self, model, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_f1': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch)
            loss = self.criterion(out, batch.y.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / max(batch_count, 1)
    
    def evaluate(self, val_loader, threshold=0.5):
        """Evaluate model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                _, preds = self.model.predict(batch, threshold=threshold)
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())
        
        # Calculate metrics
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
        
        return f1, precision, recall
    
    def train(self, train_loader, val_loader, epochs=10, save_dir='./experiments/results/phase2'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        best_f1 = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_f1, val_precision, val_recall = self.evaluate(val_loader)
            self.history['val_f1'].append(val_f1)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_model.pt'))
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"P: {val_precision:.4f} | "
                  f"R: {val_recall:.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(save_dir, 'final_model.pt'))
        
        print(f"\nTraining completed. Best F1: {best_f1:.4f}")
        return self.history

def test_trainer():
    """Test the trainer with dummy data"""
    print("Testing SimpleTrainer...")
    
    from src.models.base_gnn import SimpleGNN
    
    # Create dummy graphs
    dummy_graphs = []
    for i in range(5):
        # Create random graph data
        x = torch.randn(20, 25)  # 20 nodes, 25 features
        edge_index = torch.randint(0, 20, (2, 40))  # 40 edges
        y = torch.randint(0, 2, (20,)).float()  # Random binary labels
        
        from torch_geometric.data import Data
        graph = Data(x=x, edge_index=edge_index, y=y)
        dummy_graphs.append(graph)
    
    # Create data loaders
    train_loader = DataLoader(dummy_graphs[:3], batch_size=1, shuffle=True)
    val_loader = DataLoader(dummy_graphs[3:], batch_size=1, shuffle=False)
    
    # Create model
    model = SimpleGNN(input_dim=25, hidden_dim=32)
    
    # Create trainer (use CPU for testing)
    trainer = SimpleTrainer(model, device='cpu', lr=0.001)
    
    # Train for 2 epochs
    history = trainer.train(train_loader, val_loader, epochs=2, 
                           save_dir='./experiments/results/phase2/test')
    
    print("\n✓ SimpleTrainer test PASSED!")
    return True

if __name__ == "__main__":
    test_trainer()
