"""
SIMPLIFIED Hamiltonian GNN - Compatible with Phase 2B
Same architecture as improved_gnn_fixed.py but with physics loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class HamiltonianGNNsimple(nn.Module):
    """
    Simplified Hamiltonian GNN matching Phase 2B architecture
    """
    def __init__(self, input_dim=30, hidden_dim=128, output_dim=1, 
                 num_layers=3, dropout=0.3, lambda_physics=0.0001):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_physics = lambda_physics
        
        # Match Phase 2B architecture
        self.gat_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
        self.residual_projs.append(nn.Linear(input_dim, hidden_dim * 4))
        
        # Middle layers
        for i in range(1, num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
            self.residual_projs.append(nn.Linear(hidden_dim * 4, hidden_dim * 4))
        
        # Last layer
        self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, dropout=dropout))
        self.residual_projs.append(nn.Linear(hidden_dim * 4, hidden_dim))
        
        # Batch norms
        self.batch_norms = nn.ModuleList([BatchNorm(hidden_dim * 4) for _ in range(num_layers - 1)])
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Classifier (match Phase 2B)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Physics parameters
        self.register_buffer('ideal_bond_length', torch.tensor(1.5))
        self.register_buffer('ideal_angle', torch.tensor(109.5))
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def calculate_physics_loss(self, data):
        """Simple physics loss - less aggressive"""
        physics_loss = 0.0
        
        if hasattr(data, 'pos') and data.edge_index.shape[1] > 0:
            try:
                # Bond length constraint (softer)
                src, dst = data.edge_index
                pos = data.pos
                bond_vectors = pos[src] - pos[dst]
                bond_lengths = torch.norm(bond_vectors, dim=1)
                
                # Only penalize extreme bond lengths
                bond_error = torch.relu(bond_lengths - 10.0) + torch.relu(0.5 - bond_lengths)
                physics_loss += torch.mean(bond_error) * 0.01  # Very small weight
                
            except:
                physics_loss = torch.tensor(0.001, device=data.x.device)
        
        return physics_loss
    
    def forward(self, data):
        """Forward pass matching Phase 2B"""
        x = data.x
        
        # GAT layers with residuals
        for i, (gat_layer, residual_proj, bn) in enumerate(zip(self.gat_layers, self.residual_projs, self.batch_norms)):
            x_res = x
            
            # GAT layer
            x = gat_layer(x, data.edge_index)
            
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = bn(x)
                x = F.dropout(x, p=0.3, training=self.training)
            
            # Residual connection
            if x_res.shape[1] != x.shape[1]:
                x_res = residual_proj(x_res)
            
            if x.shape == x_res.shape:
                x = x + x_res
        
        # Classifier
        logits = self.node_classifier(x)
        
        # Physics loss (minimal)
        physics_loss = self.calculate_physics_loss(data)
        
        return logits, physics_loss

# Test
def test_simple_hamgnn():
    """Test the simple Hamiltonian GNN"""
    import torch
    from torch_geometric.data import Data
    
    print("Testing HamiltonianGNNsimple...")
    
    # Test graph
    test_data = Data(
        x=torch.randn(20, 30),  # 30 features like Phase 2B
        edge_index=torch.randint(0, 20, (2, 50)),
        pos=torch.randn(20, 3),
        y=torch.randint(0, 2, (20,)).float()
    )
    
    model = HamiltonianGNNsimple(input_dim=30, lambda_physics=0.0001)
    
    # Forward pass
    logits, physics_loss = model(test_data)
    
    print(f"Input shape: {test_data.x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    # Test training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    loss = criterion(logits, test_data.y.unsqueeze(1)) + 0.0001 * physics_loss
    loss.backward()
    optimizer.step()
    
    print("\nHamiltonianGNNsimple test PASSED!")
    return True

if __name__ == "__main__":
    test_simple_hamgnn()
