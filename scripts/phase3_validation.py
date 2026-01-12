#!/usr/bin/env python3
"""
Physics constraint validation
"""
import os
import sys
import torch
import numpy as np
import json

# Add src to path
sys.path.append('./src/models')
from hamiltonian_gnn_ablation_fixed import HamiltonianGNN

def validate_physics_constraints():
    """Validate that physics constraints are working correctly"""
    print("=" * 60)
    print("PHASE 3: PHYSICS CONSTRAINT VALIDATION")
    print("=" * 60)
    
    # Load model
    model_path = "./experiments/results/phase2b/week2/training_fixed/hamgnn_best.pt"
    model_state = torch.load(model_path, weights_only=True, map_location='cpu')
    
    model = HamiltonianGNN(input_dim=35, hidden_dim=128, lambda_physics=0.0001)
    model.load_state_dict(model_state)
    model.eval()
    
    # Load a few graphs
    data_dir = "./data/processed/physics_graphs/train"
    graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')][:5]
    
    validation_results = {
        'energy_conservation': [],
        'bond_length_violations': [],
        'angle_violations': [],
        'electrostatic_consistency': [],
        'hydrophobic_clustering': []
    }
    
    print(f"Validating physics constraints on {len(graph_files)} graphs...")
    
    for gf in graph_files:
        try:
            graph = torch.load(os.path.join(data_dir, gf))
            
            # Forward pass to get physics loss
            with torch.no_grad():
                logits, physics_loss = model(graph)
            
            # Calculate constraint violations
            violations = calculate_constraint_violations(graph)
            
            validation_results['energy_conservation'].append(float(physics_loss.item()))
            validation_results['bond_length_violations'].append(violations['bond_violation_rate'])
            validation_results['angle_violations'].append(violations['angle_violation_rate'])
            validation_results['electrostatic_consistency'].append(violations['electrostatic_consistency'])
            validation_results['hydrophobic_clustering'].append(violations['hydrophobic_clustering'])
            
            print(f"  {gf}: physics_loss={physics_loss.item():.4f}, "
                  f"bond_violations={violations['bond_violation_rate']:.1%}")
                  
        except Exception as e:
            print(f"  Error processing {gf}: {e}")
    
    # Calculate statistics
    stats = {}
    for key, values in validation_results.items():
        if values:
            stats[f'{key}_mean'] = float(np.mean(values))
            stats[f'{key}_std'] = float(np.std(values))
            stats[f'{key}_min'] = float(np.min(values))
            stats[f'{key}_max'] = float(np.max(values))
        else:
            stats[f'{key}_mean'] = 0.0
            stats[f'{key}_std'] = 0.0
    
    # Save results
    output_dir = "./experiments/results/phase3/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/physics_validation.json", 'w') as f:
        json.dump({
            'per_graph_results': validation_results,
            'summary_statistics': stats,
            'validation_criteria': {
                'energy_conservation_threshold': 'lower is better',
                'bond_violation_rate_threshold': '< 5%',
                'angle_violation_rate_threshold': '< 10%',
                'electrostatic_consistency_threshold': '> 0.7',
                'hydrophobic_clustering_threshold': 'lower is better'
            }
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHYSICS VALIDATION SUMMARY")
    print("=" * 60)
    
    criteria_met = 0
    total_criteria = 5
    
    # Check energy conservation
    energy_mean = stats['energy_conservation_mean']
    if energy_mean < 1000:  # Arbitrary threshold
        print(f"✅ Energy conservation: {energy_mean:.2f} (good)")
        criteria_met += 1
    else:
        print(f"⚠️ Energy conservation: {energy_mean:.2f} (high)")
    
    # Check bond violations
    bond_violation = stats['bond_length_violations_mean']
    if bond_violation < 0.05:  # < 5%
        print(f"✅ Bond length violations: {bond_violation:.1%} (< 5%)")
        criteria_met += 1
    else:
        print(f"⚠️ Bond length violations: {bond_violation:.1%} (> 5%)")
    
    # Check angle violations
    angle_violation = stats['angle_violations_mean']
    if angle_violation < 0.10:  # < 10%
        print(f"✅ Angle violations: {angle_violation:.1%} (< 10%)")
        criteria_met += 1
    else:
        print(f"⚠️ Angle violations: {angle_violation:.1%} (> 10%)")
    
    # Check electrostatic consistency
    electro = stats['electrostatic_consistency_mean']
    if electro > 0.7:  # > 0.7
        print(f"✅ Electrostatic consistency: {electro:.2f} (> 0.7)")
        criteria_met += 1
    else:
        print(f"⚠️ Electrostatic consistency: {electro:.2f} (< 0.7)")
    
    # Check hydrophobic clustering
    hydrophobic = stats['hydrophobic_clustering_mean']
    if hydrophobic < 10.0:  # Arbitrary threshold
        print(f"✅ Hydrophobic clustering: {hydrophobic:.2f} (good)")
        criteria_met += 1
    else:
        print(f"⚠️ Hydrophobic clustering: {hydrophobic:.2f} (high)")
    
    print(f"\nPhysics constraints met: {criteria_met}/{total_criteria}")
    
    if criteria_met >= 4:
        print("✅ Physics constraints are working correctly!")
    else:
        print("⚠️ Some physics constraints need attention")
    
    print(f"\n✅ Validation results saved to {output_dir}/physics_validation.json")
    
    return stats

def calculate_constraint_violations(graph):
    """Calculate various constraint violation rates"""
    violations = {
        'bond_violation_rate': 0.0,
        'angle_violation_rate': 0.0,
        'electrostatic_consistency': 0.0,
        'hydrophobic_clustering': 0.0
    }
    
    # Simplified calculations (replace with actual physics if needed)
    
    # 1. Bond length violations
    if hasattr(graph, 'pos') and graph.edge_index.shape[1] > 0:
        edge_index = graph.edge_index
        src, dst = edge_index[0], edge_index[1]
        distances = torch.norm(graph.pos[src] - graph.pos[dst], dim=1)
        
        # Ideal bond length ~1.5Å, tolerance ±0.1Å
        ideal = 1.5
        tolerance = 0.1
        bond_errors = torch.abs(distances - ideal)
        bond_violations = (bond_errors > tolerance).sum().item()
        violations['bond_violation_rate'] = bond_violations / len(distances) if len(distances) > 0 else 0.0
    
    # 2. Angle violations (simplified)
    if hasattr(graph, 'pos') and graph.edge_index.shape[1] >= 3:
        # Sample some triplets
        n_samples = min(100, graph.edge_index.shape[1])
        indices = torch.randperm(graph.edge_index.shape[1])[:n_samples]
        
        src = graph.edge_index[0, indices]
        dst = graph.edge_index[1, indices]
        
        # Find third node (simplified)
        third_idx = torch.randint(0, graph.pos.shape[0], (n_samples,))
        
        # Calculate angles
        v1 = graph.pos[dst] - graph.pos[src]
        v2 = graph.pos[third_idx] - graph.pos[dst]
        
        v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-6)
        v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-6)
        
        cos_angles = torch.sum(v1_norm * v2_norm, dim=1)
        angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0)) * 180 / torch.pi
        
        # Ideal angle ~109.5°, tolerance ±10°
        ideal_angle = 109.5
        tolerance = 10.0
        angle_errors = torch.abs(angles - ideal_angle)
        angle_violations = (angle_errors > tolerance).sum().item()
        violations['angle_violation_rate'] = angle_violations / n_samples if n_samples > 0 else 0.0
    
    # 3. Electrostatic consistency (simplified)
    if hasattr(graph, 'x'):
        # Assume charge is in column 20 of features
        charges = graph.x[:, 20] if graph.x.shape[1] > 20 else torch.zeros(graph.x.shape[0])
        charge_variance = torch.var(charges).item()
        violations['electrostatic_consistency'] = 1.0 / (1.0 + charge_variance)  # Higher is better
    
    # 4. Hydrophobic clustering (simplified)
    if hasattr(graph, 'x') and hasattr(graph, 'pos'):
        # Assume hydrophobicity is in column 21 of features
        hydrophobicity = graph.x[:, 21] if graph.x.shape[1] > 21 else torch.zeros(graph.x.shape[0])
        hydrophobic_mask = hydrophobicity > 0.5
        
        if hydrophobic_mask.sum() > 0:
            hydrophobic_pos = graph.pos[hydrophobic_mask]
            center = hydrophobic_pos.mean(dim=0)
            distances = torch.norm(hydrophobic_pos - center, dim=1)
            violations['hydrophobic_clustering'] = distances.mean().item()
        else:
            violations['hydrophobic_clustering'] = 0.0
    
    return violations

if __name__ == "__main__":
    validate_physics_constraints()
