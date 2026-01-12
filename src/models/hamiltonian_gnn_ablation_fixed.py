#!/usr/bin/env python3
"""
Hamiltonian GNN with component disabling - MATCHES PHASE 2B ARCHITECTURE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class HamiltonianGNNAblationFixed(nn.Module):
    """Hamiltonian GNN that matches Phase 2B architecture exactly"""
    
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=1,
                 lambda_physics=0.0001, dropout=0.3, num_layers=3,
                 disable_components=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_physics = lambda_physics
        self.disable_components = disable_components if disable_components else []
        
        print(f"Model initialized with disabled components: {self.disable_components}")
        
        # Input projection (matches Phase 2B)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers (matches Phase 2B)
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # First GAT layer: hidden_dim -> hidden_dim//4 * heads
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
        
        # Last GAT layer
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, dropout=dropout))
        
        # Batch norms
        for _ in range(num_layers):
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Residual projections
        for _ in range(num_layers):
            self.residual_projs.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Node classifier (matches Phase 2B)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Physics parameters (these won't be in saved state, that's OK)
        self.register_buffer('ideal_bond_length', torch.tensor(1.5))
        self.register_buffer('bond_tolerance', torch.tensor(0.1))
        self.register_buffer('ideal_angle', torch.tensor(109.5))
        self.register_buffer('angle_tolerance', torch.tensor(5.0))
    
    def forward(self, data):
        """
        Forward pass matching Phase 2B
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x_res = x  # For residual connection
        
        # GAT layers with residuals (matches Phase 2B exactly)
        for i in range(len(self.gat_layers)):
            # GAT layer
            x_gat = self.gat_layers[i](x, edge_index)
            
            # Apply activation and dropout
            x_gat = F.relu(x_gat)
            x_gat = F.dropout(x_gat, p=0.3, training=self.training)
            
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
        
        # Calculate physics loss if enabled
        physics_loss = torch.tensor(0.0, device=x.device)
        if self.lambda_physics > 0:
            physics_loss = self.calculate_physics_loss(data)
        
        return node_pred, physics_loss
    
    def calculate_physics_loss(self, data):
        """Calculate physics loss with component disabling"""
        total_physics_loss = torch.tensor(0.0, device=data.x.device)
        
        # Simplified physics losses (for ablation study)
        if 'electrostatics' not in self.disable_components:
            elec_loss = self.electrostatic_loss(data)
            total_physics_loss = total_physics_loss + elec_loss
        
        if 'vdw' not in self.disable_components:
            vdw_loss = self.vdw_loss(data)
            total_physics_loss = total_physics_loss + vdw_loss
        
        if 'hydrogen_bonds' not in self.disable_components:
            hbond_loss = self.hydrogen_bond_loss(data)
            total_physics_loss = total_physics_loss + hbond_loss
        
        if 'hydrophobic' not in self.disable_components:
            hydrophobic_loss = self.hydrophobic_loss(data)
            total_physics_loss = total_physics_loss + hydrophobic_loss
        
        return total_physics_loss * self.lambda_physics
    
    def electrostatic_loss(self, data):
        """Simplified electrostatic loss"""
        if not hasattr(data, 'x') or data.x.shape[1] < 21:
            return torch.tensor(0.0, device=data.x.device)
        
        # Use charge feature (assumed at column 20)
        charges = data.x[:, 20]  # Rough charge estimate
        
        if data.edge_index.shape[1] > 0:
            src, dst = data.edge_index[0], data.edge_index[1]
            q1 = charges[src]
            q2 = charges[dst]
            
            # Simple loss: opposite charges should attract, same should repel
            electrostatic = q1 * q2
            return torch.mean(torch.abs(electrostatic))
        
        return torch.tensor(0.0, device=data.x.device)
    
    def vdw_loss(self, data):
        """Simplified van der Waals loss"""
        if data.edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=data.x.device)
        
        src, dst = data.edge_index[0], data.edge_index[1]
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=1)
        
        # Penalize distances too different from ideal
        ideal = self.ideal_bond_length
        errors = torch.abs(distances - ideal)
        return torch.mean(F.relu(errors - self.bond_tolerance))
    
    def hydrogen_bond_loss(self, data):
        """Simplified hydrogen bond loss"""
        return torch.tensor(0.0, device=data.x.device)  # Skip for now
    
    def hydrophobic_loss(self, data):
        """Simplified hydrophobic loss"""
        return torch.tensor(0.0, device=data.x.device)  # Skip for now
    
    def predict_proba(self, data, threshold=0.5):
        """Predict probabilities"""
        with torch.no_grad():
            logits, _ = self.forward(data)
            probs = torch.sigmoid(logits)
            labels = (probs > threshold).float()
        
        return probs, labels

# Alias for backward compatibility
HamiltonianGNN = HamiltonianGNNAblationFixed
