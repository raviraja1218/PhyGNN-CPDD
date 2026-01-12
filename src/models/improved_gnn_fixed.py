"""
Fixed Improved GNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool

class ImprovedGNNFixed(nn.Module):
    def __init__(self, input_dim=47, hidden_dim=128, output_dim=1, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim // 2, heads=2, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        
        # Input projection
        x = self.input_proj(x)
        
        # GAT layers
        x1 = F.relu(self.gat1(x, edge_index))
        x1 = self.bn1(x1)
        
        x2 = F.relu(self.gat2(x1, edge_index))
        x2 = self.bn2(x2)
        
        x3 = self.gat3(x2, edge_index)
        x3 = self.bn3(x3)
        
        # Global context (if batched)
        if batch is not None:
            global_feat = global_mean_pool(x3, batch)
            global_feat_expanded = global_feat[batch]
            x_final = torch.cat([x3, global_feat_expanded], dim=1)
        else:
            # Single graph
            global_feat = x3.mean(dim=0, keepdim=True)
            global_feat_expanded = global_feat.expand(x3.size(0), -1)
            x_final = torch.cat([x3, global_feat_expanded], dim=1)
        
        # Classification
        out = self.classifier(x_final)
        
        return out

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.float()
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def test_fixed_gnn():
    """Test the fixed GNN"""
    print("Testing Fixed ImprovedGNN...")
    
    # Create dummy data
    x = torch.randn(100, 47)
    edge_index = torch.randint(0, 100, (2, 300))
    y = torch.zeros(100)
    y[:10] = 1.0
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Add batch attribute
    data.batch = torch.zeros(100, dtype=torch.long)
    
    # Create model
    model = ImprovedGNNFixed(input_dim=47, hidden_dim=64)
    
    # Test forward pass
    with torch.no_grad():
        output = model(data)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✓ Fixed ImprovedGNN test PASSED!")
    return True

if __name__ == "__main__":
    test_fixed_gnn()
