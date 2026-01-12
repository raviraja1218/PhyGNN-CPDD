"""
Train Hamiltonian GNN on physics-enhanced graphs
"""
import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import json
from src.models.hamiltonian_gnn import HamiltonianGNN
from src.training.physics_trainer import PhysicsTrainer

def train_hamiltonian_gnn():
    """Train Hamiltonian GNN with physics constraints"""
    print("=" * 60)
    print("HAMILTONIAN GNN TRAINING")
    print("=" * 60)
    
    # Load processed IDs
    with open('./experiments/results/phase2b/week1/processed_physics_ids.txt', 'r') as f:
        physics_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loading {len(physics_ids)} physics-enhanced graphs...")
    
    # Load all physics graphs
    graphs = []
    for pid in physics_ids:
        graph_path = f"./data/processed/physics_graphs/train/{pid}_physics.pt"
        if os.path.exists(graph_path):
            graph = torch.load(graph_path)
            graphs.append(graph)
    
    print(f"Loaded {len(graphs)} physics graphs")
    
    # Split into train/validation (same as Phase 2A: 80/20)
    n_train = int(0.8 * len(graphs))
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:]
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Validation: {len(val_graphs)} graphs")
    
    # Calculate class imbalance for weighted loss
    all_labels = torch.cat([g.y for g in train_graphs])
    pos_weight = (all_labels == 0).sum() / (all_labels == 1).sum()
    print(f"Class imbalance - Positive weight: {pos_weight:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # Get input dimension from first graph
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Hyperparameters (tune these)
    hyperparams = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'physics_weight': 0.1,  # λ parameter - tune this!
        'pos_weight': pos_weight.item() if torch.is_tensor(pos_weight) else pos_weight
    }
    
    print("\nHyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = HamiltonianGNN(
        input_dim=hyperparams['input_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        physics_weight=hyperparams['physics_weight']
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay'],
        pos_weight=hyperparams['pos_weight'],
        physics_weight=hyperparams['physics_weight']
    )
    
    print(f"\nUsing device: {trainer.device}")
    
    # Create save directory
    save_dir = "./experiments/results/phase2b/week2/training"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        early_stopping_patience=20,
        save_dir=save_dir
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'hamgnn_best.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
    
    # Evaluate on validation set
    val_total, val_pred, val_phys, val_f1, val_precision, val_recall, val_auc = \
        trainer.evaluate(val_loader)
    
    # Compare with Base GNN from Phase 2A
    base_f1 = 0.3077  # From Phase 2A
    
    print(f"\nPerformance Summary:")
    print(f"  Validation F1:      {val_f1:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall:    {val_recall:.4f}")
    print(f"  Validation AUC:       {val_auc:.4f}")
    print(f"  Prediction Loss:      {val_pred:.4f}")
    print(f"  Physics Loss:         {val_phys:.4f}")
    print(f"  Total Loss:           {val_total:.4f}")
    print(f"\nComparison with Base GNN (Phase 2A):")
    print(f"  Base GNN F1:         {base_f1:.4f}")
    print(f"  Hamiltonian GNN F1:  {val_f1:.4f}")
    print(f"  Improvement:         {val_f1 - base_f1:+.4f}")
    print(f"  Relative:            {(val_f1 - base_f1)/base_f1*100:+.1f}%")
    
    # Save final performance
    performance = {
        'hamgnn_f1': float(val_f1),
        'hamgnn_precision': float(val_precision),
        'hamgnn_recall': float(val_recall),
        'hamgnn_auc': float(val_auc),
        'hamgnn_pred_loss': float(val_pred),
        'hamgnn_physics_loss': float(val_phys),
        'hamgnn_total_loss': float(val_total),
        'base_gnn_f1': float(base_f1),
        'improvement_absolute': float(val_f1 - base_f1),
        'improvement_relative': float((val_f1 - base_f1) / base_f1 * 100),
        'physics_weight': float(hyperparams['physics_weight']),
        'num_train_graphs': len(train_graphs),
        'num_val_graphs': len(val_graphs)
    }
    
    with open(os.path.join(save_dir, 'hamgnn_performance.json'), 'w') as f:
        json.dump(performance, f, indent=2)
    
    print(f"\n✓ Performance saved to: {os.path.join(save_dir, 'hamgnn_performance.json')}")
    
    # Check if target achieved
    target_f1 = 0.35
    if val_f1 >= target_f1:
        print(f"\n🎉 TARGET ACHIEVED! F1 = {val_f1:.4f} >= {target_f1:.4f}")
        return True, performance
    else:
        print(f"\n⚠️  TARGET NOT MET: F1 = {val_f1:.4f} < {target_f1:.4f}")
        print("Consider tuning physics_weight or other hyperparameters")
        return False, performance

if __name__ == "__main__":
    success, performance = train_hamiltonian_gnn()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)
