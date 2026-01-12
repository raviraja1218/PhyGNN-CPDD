#!/usr/bin/env python3
"""
Phase 2C Optimized Training: Fix issues and achieve F1 > 0.60
"""
import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
import time

# Add src to path
sys.path.append('./src/models')
sys.path.append('./src/training')

from hamiltonian_gnn_phase2c import HamiltonianGNN
from physics_trainer import PhysicsTrainer

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_existing_graphs():
    """Load existing physics graphs from Phase 2B"""
    print("Loading existing Phase 2B physics graphs...")
    
    # Try different directories
    possible_dirs = [
        './data/processed/physics_graphs/train/',
        './data/processed/graphs_simple_enhanced/',
        './data/processed/graphs_working/train/'
    ]
    
    source_dir = None
    for graph_dir in possible_dirs:
        if os.path.exists(graph_dir):
            files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
            if files:
                print(f"  Found {len(files)} graphs in {graph_dir}")
                source_dir = graph_dir
                break
    
    if not source_dir:
        print("ERROR: No existing graphs found!")
        return None
    
    # Load graphs
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
    graphs = []
    
    for f in tqdm(files, desc="Loading graphs"):
        try:
            graph = torch.load(os.path.join(source_dir, f), weights_only=False)
            graphs.append(graph)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    
    # Split into train/val (80/20 split - more training data)
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"  Training: {len(train_graphs)} graphs")
    print(f"  Validation: {len(val_graphs)} graphs")
    
    return train_graphs, val_graphs

def optimize_hyperparameters():
    """Return optimized hyperparameters based on Phase 2B learnings"""
    
    # From Phase 2B: λ=0.0001 worked, but maybe too strong
    # Let's try different values
    hyperparam_options = [
        {
            'name': 'Very Weak Physics',
            'lambda_physics': 1e-6,  # 100x weaker
            'learning_rate': 0.001,
            'batch_size': 8,
            'hidden_dim': 128,
            'pos_weight': 20.0,
            'patience': 30  # More patience
        },
        {
            'name': 'Weak Physics',
            'lambda_physics': 1e-5,  # 10x weaker
            'learning_rate': 0.001,
            'batch_size': 8,
            'hidden_dim': 128,
            'pos_weight': 20.0,
            'patience': 30
        },
        {
            'name': 'Phase2B Setting',
            'lambda_physics': 1e-4,  # Original
            'learning_rate': 0.001,
            'batch_size': 8,
            'hidden_dim': 128,
            'pos_weight': 20.0,
            'patience': 30
        },
        {
            'name': 'Stronger Model',
            'lambda_physics': 1e-5,  # Weak physics
            'learning_rate': 0.001,
            'batch_size': 4,  # Smaller batch
            'hidden_dim': 256,  # Larger model
            'pos_weight': 20.0,
            'patience': 40
        }
    ]
    
    return hyperparam_options

def train_with_hyperparams(train_graphs, val_graphs, hyperparams):
    """Train with specific hyperparameters"""
    from torch_geometric.loader import DataLoader
    
    print(f"\nTraining with: {hyperparams['name']}")
    print(f"  λ (physics): {hyperparams['lambda_physics']}")
    print(f"  Hidden dim: {hyperparams['hidden_dim']}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, 
                             batch_size=hyperparams['batch_size'],
                             shuffle=True)
    val_loader = DataLoader(val_graphs,
                           batch_size=hyperparams['batch_size'],
                           shuffle=False)
    
    # Create model
    input_dim = train_graphs[0].x.shape[1]
    model = HamiltonianGNN(
        input_dim=input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        lambda_physics=hyperparams['lambda_physics']
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams['learning_rate'],
        pos_weight=hyperparams['pos_weight'],
        lambda_physics=hyperparams['lambda_physics']
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,  # More epochs
        early_stopping_patience=hyperparams['patience'],
        save_dir=f"./experiments/results/phase2c/{hyperparams['name'].replace(' ', '_')}",
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Get results
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    final_physics_ratio = history['physics_loss_ratio'][-1] if history['physics_loss_ratio'] else 0
    
    results = {
        'name': hyperparams['name'],
        'best_f1': float(best_f1),
        'final_physics_ratio': float(final_physics_ratio),
        'training_time_minutes': float(training_time / 60),
        'epochs_trained': len(history['train_loss']),
        'hyperparams': hyperparams,
        'phase2b_baseline': 0.5444,
        'improvement': float(best_f1 - 0.5444),
        'achieved_target': bool(best_f1 > 0.60)
    }
    
    return results, trainer.model

def run_optimized_training():
    """Run training with different hyperparameter settings"""
    print("=" * 70)
    print("PHASE 2C OPTIMIZED TRAINING")
    print("Goal: Achieve F1 > 0.60 (Phase 2B: 0.5444)")
    print("Testing different hyperparameter combinations")
    print("=" * 70)
    
    # Load graphs
    train_graphs, val_graphs = load_existing_graphs()
    if not train_graphs:
        return None
    
    # Get hyperparameter options
    hyperparam_options = optimize_hyperparameters()
    
    all_results = []
    best_model = None
    best_f1 = 0
    
    # Try each hyperparameter set
    for hyperparams in hyperparam_options:
        try:
            results, model = train_with_hyperparams(train_graphs, val_graphs, hyperparams)
            all_results.append(results)
            
            if results['best_f1'] > best_f1:
                best_f1 = results['best_f1']
                best_model = model
                print(f"\n🎯 NEW BEST: {results['name']} - F1: {results['best_f1']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error with {hyperparams['name']}: {e}")
            continue
    
    # Save all results
    results_dir = './experiments/results/phase2c/optimized_training'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    # Find best result
    if all_results:
        best_result = max(all_results, key=lambda x: x['best_f1'])
        
        print("\n" + "=" * 70)
        print("BEST RESULT:")
        print("=" * 70)
        print(f"Configuration: {best_result['name']}")
        print(f"Best F1: {best_result['best_f1']:.4f}")
        print(f"Phase 2B baseline: 0.5444")
        print(f"Improvement: {best_result['improvement']:+.4f}")
        print(f"Physics ratio: {best_result['final_physics_ratio']:.3f}")
        print(f"Training time: {best_result['training_time_minutes']:.1f} minutes")
        
        if best_result['best_f1'] > 0.60:
            print("\n🎉 TARGET ACHIEVED: F1 > 0.60!")
            print(f"   Actual: {best_result['best_f1']:.4f}")
            
            # Save best model
            torch.save(best_model.state_dict(), f'{results_dir}/best_model.pt')
            print(f"   Best model saved to: {results_dir}/best_model.pt")
        else:
            print(f"\n⚠️ Below target: {best_result['best_f1']:.4f} < 0.60")
            print(f"   Need: {0.60 - best_result['best_f1']:.4f} improvement")
        
        print("=" * 70)
        
        return best_result
    
    return None

def create_final_summary():
    """Create final summary report"""
    results_dir = './experiments/results/phase2c/optimized_training'
    results_file = f'{results_dir}/all_results.json'
    
    if not os.path.exists(results_file):
        print("No results found. Run training first.")
        return False
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Find best result
    best_result = max(all_results, key=lambda x: x['best_f1'])
    
    # Create comprehensive summary
    summary = f"""
# PHASE 2C FINAL SUMMARY

## PERFORMANCE OVERVIEW

**Best Configuration:** {best_result['name']}
**Best F1 Score:** {best_result['best_f1']:.4f}
**Target (F1 > 0.60):** {'✅ ACHIEVED' if best_result['best_f1'] > 0.60 else '❌ NOT ACHIEVED'}

## COMPARISON ACROSS PHASES

| Phase | Method | F1 Score | Improvement |
|-------|--------|----------|-------------|
| **1** | Geometric Baseline | 0.077 | - |
| **2A** | Base GNN | 0.3077 | +298% |
| **2B** | Hamiltonian GNN (70 proteins) | 0.5444 | +77% |
| **2C** | **Optimized Hamiltonian GNN** | **{best_result['best_f1']:.4f}** | **{best_result['improvement']/0.5444*100:+.1f}%** |

## HYPERPARAMETERS (Best Configuration)

```json
{json.dumps(best_result['hyperparams'], indent=2)}
