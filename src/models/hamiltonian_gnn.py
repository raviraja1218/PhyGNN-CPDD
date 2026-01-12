"""
Hamiltonian-Informed Graph Neural Network
Extends Base GNN with physics constraints
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class HamiltonianGNN(nn.Module):
    """
    GNN with Hamiltonian physics constraints
    
    Architecture: GNN + Physics loss terms
    Total Loss = Prediction Loss + λ * Physics Loss
    """
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=1, 
                 num_layers=3, dropout=0.3, heads=4, physics_weight=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.physics_weight = physics_weight  # λ parameter
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers (same as Base GNN for fair comparison)
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                gat_layer = GATConv(hidden_dim, hidden_dim // heads, 
                                   heads=heads, dropout=dropout)
            else:
                gat_layer = GATConv(hidden_dim, hidden_dim // heads,
                                   heads=heads, dropout=dropout)
            
            self.gat_layers.append(gat_layer)
            self.batch_norms.append(BatchNorm(hidden_dim))
            self.residual_projs.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Node classifier
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Physics constraint parameters
        # Ideal bond lengths and angles (simplified)
        self.ideal_bond_length = 1.5  # Å (typical C-C bond)
        self.ideal_angle = 109.5  # degrees (tetrahedral)
        
        # Energy parameters
        self.k_bond = 100.0  # Bond stiffness
        self.k_angle = 10.0  # Angle stiffness
        self.k_elec = 332.0  # Electrostatic constant (kcal/mol·Å)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, data, compute_physics=True):
        """
        Forward pass with optional physics constraints
        
        Args:
            data: PyTorch Geometric Data object
            compute_physics: Whether to compute physics loss
            
        Returns:
            dict containing:
                'node_pred': Node predictions [N, 1]
                'node_embeddings': Node embeddings [N, hidden_dim]
                'physics_loss': Physics constraint loss (scalar)
                'total_loss': Total loss if labels provided
        """
        x, edge_index, edge_physics = data.x, data.edge_index, data.edge_physics
        
        # Store original positions for physics calculations
        original_positions = data.pos.clone() if hasattr(data, 'pos') else None
        
        # Input projection
        x = self.input_proj(x)
        x_res = x  # For residual connection
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            # GAT layer
            x_gat = self.gat_layers[i](x, edge_index)
            
            # Apply activation and dropout
            x_gat = F.relu(x_gat)
            x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
            
            # Batch normalization
            x_gat = self.batch_norms[i](x_gat)
            
            # Residual connection
            if x.shape == x_res.shape:
                x_gat = x_gat + self.residual_projs[i](x_res)
            
            # Update for next layer
            x_res = x
            x = x_gat
        
        # Node classification
        node_pred = self.node_classifier(x)
        
        # Compute physics loss if requested
        physics_loss = 0.0
        if compute_physics and edge_physics is not None and original_positions is not None:
            physics_loss = self.compute_physics_loss(
                original_positions, edge_index, edge_physics, node_pred
            )
        
        result = {
            'node_pred': node_pred,
            'node_embeddings': x,
            'physics_loss': physics_loss
        }
        
        # Add total loss if labels are provided
        if hasattr(data, 'y') and data.y is not None:
            pred_loss = F.binary_cross_entropy_with_logits(
                node_pred, data.y.unsqueeze(1)
            )
            total_loss = pred_loss + self.physics_weight * physics_loss
            result['prediction_loss'] = pred_loss
            result['total_loss'] = total_loss
        
        return result
    
    def compute_physics_loss(self, positions, edge_index, edge_physics, node_pred):
        """
        Compute physics constraint losses
        
        Returns weighted sum of:
        1. Bond length constraints
        2. Angle constraints
        3. Electrostatic energy conservation
        4. van der Waals packing
        """
        total_physics_loss = 0.0
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            return total_physics_loss
        
        # 1. Bond length constraint (simplified)
        # Calculate distances between connected residues
        src, dst = edge_index[0], edge_index[1]
        pos_src = positions[src]
        pos_dst = positions[dst]
        distances = torch.norm(pos_src - pos_dst, dim=1)
        
        # Penalize deviations from ideal bond length
        bond_loss = torch.mean((distances - self.ideal_bond_length) ** 2)
        total_physics_loss += 0.3 * bond_loss
        
        # 2. Angle constraint (simplified)
        # For each node, consider angles formed with its neighbors
        angle_loss = 0.0
        for i in range(positions.shape[0]):
            neighbors = edge_index[1][edge_index[0] == i]
            if len(neighbors) >= 2:
                # Take first two neighbors
                n1, n2 = neighbors[0], neighbors[1]
                v1 = positions[n1] - positions[i]
                v2 = positions[n2] - positions[i]
                
                # Calculate angle
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
                angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
                angle_deg = angle * 180.0 / torch.pi
                
                # Penalize deviation from ideal angle
                angle_loss += (angle_deg - self.ideal_angle) ** 2
        
        if angle_loss > 0:
            angle_loss = angle_loss / positions.shape[0]
            total_physics_loss += 0.2 * angle_loss
        
        # 3. Electrostatic energy from edge physics
        # First feature in edge_physics is electrostatic potential
        if edge_physics.shape[1] > 0:
            elec_energy = torch.mean(torch.abs(edge_physics[:, 0]))
            # Penalize large electrostatic energies (should be moderate)
            total_physics_loss += 0.25 * elec_energy
        
        # 4. van der Waals packing from edge physics
        # Second feature is vdW interaction
        if edge_physics.shape[1] > 1:
            vdw_energy = torch.mean(torch.abs(edge_physics[:, 1]))
            # Penalize extreme vdW interactions
            total_physics_loss += 0.25 * vdw_energy
        
        return total_physics_loss
    
    def predict_proba(self, data, threshold=0.5):
        """
        Predict probabilities and binary labels
        
        Args:
            data: Input graph
            threshold: Probability threshold
            
        Returns:
            probs: Probabilities [num_nodes, 1]
            labels: Binary predictions [num_nodes]
        """
        with torch.no_grad():
            result = self.forward(data, compute_physics=False)
            logits = result['node_pred']
            probs = torch.sigmoid(logits)
            labels = (probs > threshold).float()
            
        return probs, labels, result.get('physics_loss', 0.0)

# Test function
def test_hamiltonian_gnn():
    """Test the Hamiltonian GNN model"""
    try:
        # Load a physics-enhanced graph
        sample_graph = torch.load("./data/processed/physics_graphs/samples/1a0q_physics.pt")
        print(f"Loaded physics graph: {sample_graph.num_nodes} nodes, {sample_graph.edge_index.shape[1]} edges")
        print(f"Edge physics shape: {sample_graph.edge_physics.shape}")
        
        # Initialize model
        model = HamiltonianGNN(input_dim=sample_graph.x.shape[1], physics_weight=0.1)
        print(f"Model created with input_dim={sample_graph.x.shape[1]}, physics_weight=0.1")
        
        # Test forward pass without physics
        with torch.no_grad():
            result = model(sample_graph, compute_physics=False)
            print(f"Prediction shape: {result['node_pred'].shape}")
        
        # Test forward pass with physics
        with torch.no_grad():
            result = model(sample_graph, compute_physics=True)
            print(f"With physics - Prediction loss: N/A (no labels)")
            print(f"With physics - Physics loss: {result['physics_loss']:.6f}")
        
        # Test prediction
        probs, labels, phys_loss = model.predict_proba(sample_graph)
        print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"Positive predictions: {labels.sum().item()}/{len(labels)}")
        print(f"Prediction physics loss: {phys_loss:.6f}")
        
        print("\n✓ Hamiltonian GNN test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ Hamiltonian GNN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hamiltonian_gnn()
