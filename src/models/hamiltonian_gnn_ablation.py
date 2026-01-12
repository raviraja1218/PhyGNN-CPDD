#!/usr/bin/env python3
"""
Hamiltonian GNN with component disabling for ablation study
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HamiltonianGNNAblation(nn.Module):
    """Hamiltonian GNN with ability to disable specific physics components"""
    
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=1,
                 lambda_physics=0.0001, dropout=0.3,
                 disable_components=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_physics = lambda_physics
        
        # Track which components to disable
        self.disable_components = disable_components if disable_components else []
        print(f"Disabled components: {self.disable_components}")
        
        # Linear layers for node features
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Physics constraint parameters
        self.register_buffer('ideal_bond_length', torch.tensor(1.5))  # Å
        self.register_buffer('bond_tolerance', torch.tensor(0.1))     # Å
        self.register_buffer('ideal_angle', torch.tensor(109.5))      # degrees
        self.register_buffer('angle_tolerance', torch.tensor(5.0))    # degrees
        
    def forward(self, data):
        """
        Forward pass with optional physics constraints
        
        Args:
            data: PyG Data object with x, edge_index, pos, etc.
            
        Returns:
            logits: Node predictions
            physics_loss: Physics constraint loss
        """
        x, edge_index = data.x, data.edge_index
        
        # Node feature processing
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        logits = self.classifier(x)
        
        # Calculate physics loss based on enabled components
        physics_loss = torch.tensor(0.0, device=x.device)
        
        if self.lambda_physics > 0:
            physics_loss = self.calculate_physics_loss(data)
        
        return logits, physics_loss
    
    def calculate_physics_loss(self, data):
        """Calculate physics loss with component disabling"""
        total_physics_loss = torch.tensor(0.0, device=data.x.device)
        
        # 1. Electrostatic interactions (if enabled)
        if 'electrostatics' not in self.disable_components:
            elec_loss = self.electrostatic_loss(data)
            total_physics_loss = total_physics_loss + elec_loss
        
        # 2. Van der Waals interactions (if enabled)
        if 'vdw' not in self.disable_components:
            vdw_loss = self.vdw_loss(data)
            total_physics_loss = total_physics_loss + vdw_loss
        
        # 3. Hydrogen bond potential (if enabled)
        if 'hydrogen_bonds' not in self.disable_components:
            hbond_loss = self.hydrogen_bond_loss(data)
            total_physics_loss = total_physics_loss + hbond_loss
        
        # 4. Hydrophobic interactions (if enabled)
        if 'hydrophobic' not in self.disable_components:
            hydrophobic_loss = self.hydrophobic_loss(data)
            total_physics_loss = total_physics_loss + hydrophobic_loss
        
        return total_physics_loss
    
    def electrostatic_loss(self, data):
        """Electrostatic interaction loss"""
        # Simplified: penalize opposite charges far apart, same charges close together
        if not hasattr(data, 'partial_charges'):
            return torch.tensor(0.0, device=data.x.device)
        
        # Use partial charges if available, otherwise estimate from residue type
        charges = data.partial_charges if hasattr(data, 'partial_charges') else data.x[:, 20]  # Rough estimate
        
        # Calculate pairwise charge interactions
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]
        
        q1 = charges[src]
        q2 = charges[dst]
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=1)
        
        # Coulomb-like interaction: q1*q2 / (r + epsilon)
        epsilon = 1e-6
        electrostatic = q1 * q2 / (distances + epsilon)
        
        # Loss: minimize inappropriate electrostatic interactions
        loss = torch.mean(torch.abs(electrostatic))
        return loss
    
    def vdw_loss(self, data):
        """Van der Waals interaction loss"""
        # Simplified: penalize atoms too close or too far
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]
        
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=1)
        ideal_distance = self.ideal_bond_length
        tolerance = self.bond_tolerance
        
        # Loss: distance should be close to ideal
        distance_error = torch.abs(distances - ideal_distance)
        vdw_loss = torch.mean(F.relu(distance_error - tolerance))
        
        return vdw_loss
    
    def hydrogen_bond_loss(self, data):
        """Hydrogen bond potential loss"""
        # Simplified: encourage proper H-bond geometry
        if data.edge_index.shape[1] < 3:
            return torch.tensor(0.0, device=data.x.device)
        
        # For each edge, check angle with neighboring edges
        edge_index = data.edge_index
        
        # Simple implementation: penalize sharp angles
        if edge_index.shape[1] >= 3:
            # Get random triplets
            idx = torch.randperm(edge_index.shape[1])[:min(100, edge_index.shape[1])]
            
            src = edge_index[0, idx]
            mid = edge_index[1, idx]
            
            # Find third node connected to mid
            # Simplified: use random other node
            other_idx = torch.randint(0, data.pos.shape[0], (len(idx),))
            
            # Calculate angles
            v1 = data.pos[mid] - data.pos[src]
            v2 = data.pos[other_idx] - data.pos[mid]
            
            # Normalize
            v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-6)
            v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-6)
            
            # Cosine similarity
            cos_angle = torch.sum(v1_norm * v2_norm, dim=1)
            angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)) * 180 / torch.pi
            
            # Ideal H-bond angle ~180 degrees
            ideal_angle = self.ideal_angle
            angle_error = torch.abs(angles - ideal_angle)
            
            hbond_loss = torch.mean(angle_error) / 180.0  # Normalize
            
            return hbond_loss
        
        return torch.tensor(0.0, device=data.x.device)
    
    def hydrophobic_loss(self, data):
        """Hydrophobic interaction loss"""
        # Simplified: hydrophobic residues should cluster
        if not hasattr(data, 'hydrophobicity'):
            return torch.tensor(0.0, device=data.x.device)
        
        hydrophobicity = data.hydrophobicity if hasattr(data, 'hydrophobicity') else data.x[:, 21]  # Rough estimate
        
        # Get hydrophobic residues (value > threshold)
        hydrophobic_mask = hydrophobicity > 0.5
        if hydrophobic_mask.sum() == 0:
            return torch.tensor(0.0, device=data.x.device)
        
        # Calculate center of hydrophobic cluster
        hydrophobic_pos = data.pos[hydrophobic_mask]
        if len(hydrophobic_pos) == 0:
            return torch.tensor(0.0, device=data.x.device)
        
        center = hydrophobic_pos.mean(dim=0)
        
        # Loss: hydrophobic residues should be close to center
        distances = torch.norm(hydrophobic_pos - center, dim=1)
        hydrophobic_loss = torch.mean(distances)
        
        return hydrophobic_loss
    
    def predict_proba(self, data, threshold=0.5):
        """Predict probabilities"""
        with torch.no_grad():
            logits, _ = self.forward(data)
            probs = torch.sigmoid(logits)
            labels = (probs > threshold).float()
            
        return probs, labels

# For backward compatibility
HamiltonianGNN = HamiltonianGNNAblation
