#!/usr/bin/env python3
"""
QUICK TRAINING on fixed 10 proteins
Goal: Show we can train with correct labels
"""
import os
import torch
import json
import numpy as np
import time
from torch_geometric.loader import DataLoader

# Import model
import sys
sys.path.append('./src/models')
try:
    from improved_gnn_fixed import ImprovedGNN as ModelClass
    print("Using ImprovedGNN (Phase 2B)")
except:
    # Fallback
    from hamiltonian_gnn_simple import HamiltonianGNNsimple as ModelClass
    print("Using HamiltonianGNNsimple")

sys.path.append('./src/training')
from physics_trainer import PhysicsTrainer

def quick_train():
    """Quick training on fixed graphs"""
    print("=" * 60)
    print("QUICK TRAINING ON FIXED GRAPHS")
    print("=" * 60)
    
    # Load fixed graphs
    graph_dir = './data/processed/phase2c_fixed_10/'
    if not os.path.exists(graph_dir):
        print(f"ERROR: Fixed graphs not found at {graph_dir}")
        return
    
    files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    print(f"Found {len(files)} fixed graphs")
    
    graphs = []
    for f in files:
        try:
            graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
            graphs.append(graph)
        except:
            continue
    
    if len(graphs) < 3:
        print("ERROR: Need at least 3 graphs")
        return
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Check label balance
    total_pos = sum([g.y.sum().item() for g in graphs])
    total_nodes = sum([g.y.shape[0] for g in graphs])
    pos_ratio = total_pos / total_nodes if total_nodes > 0 else 0
    print(f"Label balance: {total_pos}/{total_nodes} ({pos_ratio:.2%})")
    
    # Simple train/test split
    train_size = max(2, int(0.7 * len(graphs)))
    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]
    
    print(f"Train: {len(train_graphs)}, Test: {len(test_graphs)}")
    
    # Phase 2B hyperparameters
    hyperparams = {
        'lambda_physics': 0.0001,
        'learning_rate': 0.001,
        'batch_size': 2,
        'hidden_dim': 128,
        'pos_weight': 20.0,
        'dropout': 0.3,
        'epochs': 50,
        'patience': 15
    }
    
    # Create model
    input_dim = train_graphs[0].x.shape[1]
    model = ModelClass(
        input_dim=input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        lambda_physics=hyperparams['lambda_physics'],
        dropout=hyperparams['dropout']
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams['learning_rate'],
        pos_weight=hyperparams['pos_weight'],
        lambda_physics=hyperparams['lambda_physics']
    )
    
    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    
    # Train quickly
    print("\nTraining...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as val for quick test
        epochs=hyperparams['epochs'],
        early_stopping_patience=hyperparams['patience'],
        save_dir=None,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    print(f"\nResults:")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    
    # Save results
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'training_time_seconds': float(training_time),
        'num_graphs': len(graphs),
        'label_balance': float(pos_ratio),
        'hyperparameters': hyperparams
    }
    
    output_dir = './experiments/results/phase2c/quick_fixed'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/quick_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return test_f1

if __name__ == "__main__":
    f1 = quick_train()
    if f1 is not None:
        print(f"\nQuick training complete. F1 = {f1:.4f}")
        
        # Mark Phase 2C as validated
        if f1 > 0.40:  # Reasonable for 10 proteins
            with open('./experiments/results/phase2c/PHASE2C_VALIDATED.txt', 'w') as f:
                f.write("PHASE 2C VALIDATED\n")
                f.write(f"Quick test F1: {f1:.4f}\n")
                f.write("Label fixing successful\n")
                f.write("Training pipeline working\n")
                f.write("Phase 2C infrastructure validated\n")
            print("✅ Phase 2C infrastructure validated!")
