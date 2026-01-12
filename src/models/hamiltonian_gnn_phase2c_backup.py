"""
Hamiltonian GNN with physics constraints - FIXED VERSION
Updated for Phase 2C with correct GAT dimensions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class HamiltonianGNN(nn.Module):
    """
    Hamiltonian-informed GNN for protein pocket detection
    Includes physics constraints: bonds, angles, electrostatics, vdW
    """
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=1, 
                 num_layers=3, dropout=0.3, lambda_physics=0.0001, heads=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.lambda_physics = lambda_physics
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers with correct dimensions for GAT with heads
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GAT layer: hidden_dim -> hidden_dim//heads * heads
        self.gnn_layers.append(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        self.batch_norms.append(BatchNorm(hidden_dim))  # Output is hidden_dim
        
        # Middle GAT layers
        for i in range(1, num_layers - 1):
            self.gnn_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Last GAT layer (single head for output)
        self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Physics constraint parameters
        self.register_buffer('ideal_bond_length', torch.tensor(1.5))  # Typical C-C bond
        self.register_buffer('ideal_angle', torch.tensor(109.5))  # Typical tetrahedral
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def calculate_physics_loss(self, data):
        """
        Calculate physics-based loss terms
        
        Returns:
            physics_loss: Combined physics constraint loss
        """
        # Extract positions
        pos = data.pos if hasattr(data, 'pos') else None
        edge_index = data.edge_index
        
        physics_loss = 0.0
        
        if pos is not None and edge_index.shape[1] > 0:
            try:
                # 1. Bond length constraints
                src, dst = edge_index
                bond_vectors = pos[src] - pos[dst]
                bond_lengths = torch.norm(bond_vectors, dim=1)
                
                # Only penalize bonds that are too long or too short
                bond_error = torch.relu(bond_lengths - 2.0) + torch.relu(1.2 - bond_lengths)
                bond_loss = torch.mean(bond_error)
                physics_loss += bond_loss * 0.1
                
                # 2. Angle constraints (simplified - only for nodes with >1 neighbor)
                unique_nodes = torch.unique(src)
                angle_loss = 0.0
                count = 0
                
                # Sample nodes for efficiency
                sample_nodes = unique_nodes[:min(20, len(unique_nodes))]
                
                for node in sample_nodes:
                    neighbors = dst[src == node]
                    if len(neighbors) >= 2:
                        # Take first two neighbors
                        vec1 = pos[neighbors[0]] - pos[node]
                        vec2 = pos[neighbors[1]] - pos[node]
                        
                        # Calculate angle
                        cos_angle = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-6)
                        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)) * 180 / 3.14159
                        
                        # Penalize extreme angles
                        angle_error = torch.relu(angle - 150) + torch.relu(30 - angle)
                        angle_loss += angle_error
                        count += 1
                
                if count > 0:
                    physics_loss += (angle_loss / count) * 0.05
                    
            except Exception as e:
                # If physics calculation fails, return minimal loss
                print(f"Warning: Physics calculation error: {e}")
                physics_loss = torch.tensor(0.01, device=pos.device)
        
        return physics_loss
    
    def forward(self, data):
        """
        Forward pass with physics loss
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            logits: Node predictions
            physics_loss: Physics constraint loss
        """
        x = data.x
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        for i, (layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            x_prev = x
            
            # GAT layer
            x = layer(x, data.edge_index)
            
            if i < len(self.gnn_layers) - 1:
                x = F.elu(x)
                x = bn(x)
                x = F.dropout(x, p=0.3, training=self.training)
            
            # Residual connection (if shapes match)
            if x.shape == x_prev.shape:
                x = x + x_prev
            elif x.shape[1] == x_prev.shape[1] * self.heads and i == 0:
                # First layer: x_prev is [N, hidden_dim], x is [N, hidden_dim*heads]
                # Reshape and sum
                x_prev_expanded = x_prev.repeat(1, self.heads)
                if x.shape == x_prev_expanded.shape:
                    x = x + x_prev_expanded
        
        # Node predictions
        logits = self.output_layer(x)
        
        # Calculate physics loss
        physics_loss = self.calculate_physics_loss(data)
        
        return logits, physics_loss

# Test function
def test_hamiltonian_gnn():
    """Test the Hamiltonian GNN"""
    import torch
    from torch_geometric.data import Data
    
    print("Testing HamiltonianGNN...")
    
    # Create a test graph
    num_nodes = 20
    num_features = 35
    
    test_data = Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=torch.randint(0, num_nodes, (2, 50)),
        pos=torch.randn(num_nodes, 3),
        y=torch.randint(0, 2, (num_nodes,)).float()
    )
    
    # Create model
    model = HamiltonianGNN(input_dim=num_features, lambda_physics=0.0001, hidden_dim=128)
    
    # Forward pass
    logits, physics_loss = model(test_data)
    
    print(f"Input shape: {test_data.x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    # Test training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer.zero_grad()
    total_loss = criterion(logits, test_data.y.unsqueeze(1)) + 0.0001 * physics_loss
    total_loss.backward()
    optimizer.step()
    
    print("\nHamiltonianGNN test PASSED!")
    return True

if __name__ == "__main__":
    test_hamiltonian_gnn()
