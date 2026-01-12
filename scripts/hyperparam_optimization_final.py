#!/usr/bin/env python3
"""
FINAL: Systematic hyperparameter optimization for Phase 2C
"""
import os
import torch
import json
import numpy as np
import optuna
from torch_geometric.loader import DataLoader

sys.path.append('./src/models')
sys.path.append('./src/training')
from hamiltonian_gnn_simple import HamiltonianGNNsimple
from physics_trainer import PhysicsTrainer

def load_sample_data():
    """Load sample of data for optimization"""
    train_dir = './data/processed/physics_graphs_full/train/'
    files = os.listdir(train_dir)[:100]  # Use 100 for optimization
    
    graphs = []
    for f in files:
        if f.endswith('.pt'):
            graphs.append(torch.load(os.path.join(train_dir, f)))
    
    print(f"Loaded {len(graphs)} graphs for optimization")
    return graphs

def objective(trial):
    """Optuna objective function"""
    # Hyperparameters to optimize
    lambda_physics = trial.suggest_float('lambda_physics', 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    pos_weight = trial.suggest_float('pos_weight', 5.0, 30.0)
    
    # Load data
    graphs = load_sample_data()
    if len(graphs) < 20:
        return 0.0
    
    # Split
    train_size = int(0.8 * len(graphs))
    train_loader = DataLoader(graphs[:train_size], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(graphs[train_size:], batch_size=batch_size, shuffle=False)
    
    # Model
    input_dim = graphs[0].x.shape[1]
    model = HamiltonianGNNsimple(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lambda_physics=lambda_physics,
        dropout=dropout
    )
    
    # Trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        lambda_physics=lambda_physics
    )
    
    # Quick training (10 epochs)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        early_stopping_patience=5,
        save_dir=None,
        verbose=False
    )
    
    # Return best validation F1
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    return best_f1

def run_optimization():
    """Run systematic optimization"""
    print("=" * 60)
    print("PHASE 2C: SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='phygnn_phase2c',
        storage=f'sqlite:///./experiments/results/phase2c/optimization.db',
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=100, n_jobs=1)
    
    print(f"\nBest trial:")
    print(f"  Value (F1): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save results
    os.makedirs('./experiments/results/phase2c/hyperparameter_optimization', exist_ok=True)
    
    # Best parameters
    best_params = {
        'best_f1': study.best_value,
        'best_params': study.best_params,
        'phase2b_baseline': 0.5444,
        'improvement': study.best_value - 0.5444
    }
    
    with open('./experiments/results/phase2c/hyperparameter_optimization/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # All trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state)
        })
    
    with open('./experiments/results/phase2c/hyperparameter_optimization/all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    print(f"\nResults saved to: ./experiments/results/phase2c/hyperparameter_optimization/")
    return study.best_params

if __name__ == "__main__":
    run_optimization()
