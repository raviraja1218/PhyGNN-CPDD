#!/usr/bin/env python3
"""
EMERGENCY PHASE 2C BOOST
If regular training doesn't reach F1 > 0.60
Strategy: Combine Phase 2B success with data augmentation
"""
import os
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.append('./src/models')
sys.path.append('./src/training')
from hamiltonian_gnn_phase2c import HamiltonianGNN
from physics_trainer import PhysicsTrainer

def augment_graphs(graphs):
    """Simple data augmentation by adding noise"""
    augmented = []
    for graph in graphs:
        augmented.append(graph)
        
        # Create augmented version with noise
        if torch.rand(1) > 0.5:  # 50% chance
            aug_graph = graph.clone()
            # Add small noise to features
            noise = torch.randn_like(aug_graph.x) * 0.05
            aug_graph.x = aug_graph.x + noise
            # Add small noise to positions
            if hasattr(aug_graph, 'pos'):
                pos_noise = torch.randn_like(aug_graph.pos) * 0.1
                aug_graph.pos = aug_graph.pos + pos_noise
            augmented.append(aug_graph)
    
    print(f"Augmented {len(graphs)} graphs to {len(augmented)} graphs")
    return augmented

def emergency_boost():
    """Emergency training to reach F1 > 0.60"""
    print("=" * 60)
    print("EMERGENCY PHASE 2C BOOST")
    print("=" * 60)
    
    # Load existing physics graphs
    graph_dir = './data/processed/physics_graphs/train/'
    graphs = []
    for f in os.listdir(graph_dir):
        if f.endswith('.pt'):
            graphs.append(torch.load(os.path.join(graph_dir, f)))
    
    print(f"Loaded {len(graphs)} physics graphs")
    
    # Augment data
    augmented_graphs = augment_graphs(graphs)
    
    # Split
    train_size = int(0.85 * len(augmented_graphs))  # More training data
    train_graphs = augmented_graphs[:train_size]
    val_graphs = augmented_graphs[train_size:]
    
    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)  # Even smaller batch
    val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)
    
    # Use Phase 2B model as starting point
    phase2b_model = './experiments/results/phase2b/week2/training_fixed/hamgnn_best.pt'
    
    if os.path.exists(phase2b_model):
        print("Loading Phase 2B model as starting point...")
        model = HamiltonianGNN(input_dim=graphs[0].x.shape[1], lambda_physics=0.00003)
        model.load_state_dict(torch.load(phase2b_model))
        print("Loaded Phase 2B model (F1=0.5444)")
    else:
        print("Training from scratch...")
        model = HamiltonianGNN(input_dim=graphs[0].x.shape[1], 
                             hidden_dim=256,
                             lambda_physics=0.00003,
                             dropout=0.1)  # Less dropout
    
    # Aggressive training
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.0002,  # Very low learning rate
        pos_weight=12.0,  # Adjusted weight
        lambda_physics=0.00003  # Minimal physics
    )
    
    # Train for many epochs
    print("\nStarting aggressive training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=300,  # Many epochs
        early_stopping_patience=40,  # More patience
        save_dir='./experiments/results/phase2c/emergency_boost',
        verbose=True
    )
    
    # Get best F1
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    print(f"\nEmergency training complete. Best F1: {best_f1:.4f}")
    
    # Save results
    results = {
        'best_f1': float(best_f1),
        'phase2b_baseline': 0.5444,
        'improvement': float(best_f1 - 0.5444),
        'achieved_target': bool(best_f1 > 0.60),
        'strategy': 'emergency_boost_augmentation',
        'num_original_graphs': len(graphs),
        'num_augmented_graphs': len(augmented_graphs)
    }
    
    os.makedirs('./experiments/results/phase2c/emergency_boost', exist_ok=True)
    with open('./experiments/results/phase2c/emergency_boost/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if best_f1 > 0.60:
        print("\n🎉 EMERGENCY SUCCESS! F1 > 0.60 achieved!")
        return True
    else:
        print(f"\n❌ Still below target. F1 = {best_f1:.4f}")
        return False

if __name__ == "__main__":
    success = emergency_boost()
    
    if success:
        # Create success flag
        with open('./experiments/results/phase2c/PHASE2C_SUCCESS.txt', 'w') as f:
            f.write("PHASE 2C COMPLETE VIA EMERGENCY BOOST\n")
            f.write("Target: F1 > 0.60\n")
            f.write("Status: SUCCESS\n")
        print("\nPhase 2C marked as complete.")
    else:
        print("\nPhase 2C target not achieved. Consider next steps.")
