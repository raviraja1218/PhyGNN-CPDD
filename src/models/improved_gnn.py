"""
Improved GNN with Better Architecture for Imbalanced Data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool

class ImprovedGNN(nn.Module):
    def __init__(self, input_dim=47, hidden_dim=128, output_dim=1, dropout=0.3):
        super().__init__()
        
        # Enhanced input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Multiple GAT layers with skip connections
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.gat3 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.gat4 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.bn4 = BatchNorm(hidden_dim)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Multi-scale classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Skip connection projections
        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip3 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x0 = x  # Save for skip connections
        
        # Layer 1
        x1 = self.gat1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)
        x1 = x1 + self.skip1(x0)  # Skip connection
        
        # Layer 2
        x2 = self.gat2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = self.bn2(x2)
        x2 = x2 + self.skip2(x1)  # Skip connection
        
        # Layer 3
        x3 = self.gat3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = self.bn3(x3)
        x3 = x3 + self.skip3(x2)  # Skip connection
        
        # Layer 4
        x4 = self.gat4(x3, edge_index)
        x4 = self.bn4(x4)
        
        # Multi-scale feature concatenation
        x_multi = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Attention pooling for graph-level context
        if hasattr(data, 'batch'):
            batch = data.batch
            # Global context
            global_feat = global_mean_pool(x_multi, batch)
            # Broadcast global features to nodes
            global_feat_expanded = global_feat[batch]
            # Concatenate node features with global context
            x_final = torch.cat([x_multi, global_feat_expanded], dim=1)
        else:
            # If no batch, use mean of all nodes as global context
            global_feat = x_multi.mean(dim=0, keepdim=True)
            global_feat_expanded = global_feat.expand(x_multi.size(0), -1)
            x_final = torch.cat([x_multi, global_feat_expanded], dim=1)
        
        # Classification
        out = self.classifier(x_final)
        
        return out
    
    def predict(self, data, threshold=0.5):
        """Make predictions with calibrated threshold"""
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.sigmoid(logits)
            
            # For imbalanced data, we might want to adjust threshold
            # Based on validation performance
            preds = (probs > threshold).float()
            
            return probs, preds

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Convert targets to float
        targets = targets.float()
        
        # Calculate pt
        pt = torch.exp(-BCE_loss)
        
        # Focal loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def test_improved_gnn():
    """Test the improved GNN"""
    print("Testing ImprovedGNN...")
    
    # Create dummy data
    x = torch.randn(100, 47)  # 100 nodes, 47 features
    edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
    y = torch.zeros(100)
    y[:10] = 1.0  # 10% positive
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create model
    model = ImprovedGNN(input_dim=47, hidden_dim=64)
    
    # Test forward pass
    with torch.no_grad():
        output = model(data)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test focal loss
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(output, y.unsqueeze(1))
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("\n✓ ImprovedGNN test PASSED!")
    return True

if __name__ == "__main__":
    test_improved_gnn()
