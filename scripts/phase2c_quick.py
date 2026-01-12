#!/usr/bin/env python3
"""
Phase 2C Quick Execution: Build on Phase 2B success
Goal: Achieve F1 > 0.60 using existing 70 physics graphs
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

def load_existing_graphs():
    """Load existing physics graphs from Phase 2B"""
    print("Loading existing Phase 2B physics graphs...")
    
    # Check where the graphs are
    possible_dirs = [
        './data/processed/physics_graphs/train/',
        './data/processed/graphs_simple_enhanced/',
        './data/processed/graphs_working/train/'
    ]
    
    graphs = []
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
    
    # Load first 70 graphs (or all available)
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])[:70]
    
    for f in tqdm(files, desc="Loading graphs"):
        try:
            graph = torch.load(os.path.join(source_dir, f), weights_only=False)
            graphs.append(graph)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    
    # Split into train/val (70/30 split)
    train_size = int(0.7 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"  Training: {len(train_graphs)} graphs")
    print(f"  Validation: {len(val_graphs)} graphs")
    
    return train_graphs, val_graphs

def run_quick_training():
    """Quick training to reach F1 > 0.60"""
    print("=" * 60)
    print("PHASE 2C QUICK EXECUTION")
    print("Goal: Achieve F1 > 0.60 using Phase 2B success")
    print("Phase 2B baseline: F1 = 0.5444")
    print("=" * 60)
    
    # Load existing graphs
    train_graphs, val_graphs = load_existing_graphs()
    if not train_graphs:
        return
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Check feature dimensions
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # Hyperparameters from Phase 2B discovery
    hyperparams = {
        'lambda_physics': 0.0001,  # Critical: From Phase 2B
        'learning_rate': 0.001,
        'batch_size': 8,
        'hidden_dim': 128,
        'pos_weight': 20.0  # For 4.6% class imbalance
    }
    
    print("\nUsing optimized hyperparameters from Phase 2B:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Create model
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
    print("\nStarting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_dir='./experiments/results/phase2c/quick_training',
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Get best F1
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    # Create results
    results = {
        'best_f1': best_f1,
        'phase2b_baseline': 0.5444,
        'improvement': best_f1 - 0.5444,
        'hyperparameters': hyperparams,
        'training_time_minutes': training_time / 60,
        'num_training_graphs': len(train_graphs),
        'num_validation_graphs': len(val_graphs),
        'achieved_target': best_f1 > 0.60
    }
    
    # Save results
    os.makedirs('./experiments/results/phase2c/quick_training', exist_ok=True)
    with open('./experiments/results/phase2c/quick_training/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Phase 2B baseline: 0.5444")
    print(f"Improvement: {best_f1 - 0.5444:+.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    if best_f1 > 0.60:
        print("🎉 SUCCESS: Achieved target F1 > 0.60!")
        print(f"   Actual: {best_f1:.4f}")
    else:
        print("⚠️ Below target F1 > 0.60")
        print(f"   Current: {best_f1:.4f}, Target: 0.60")
        print(f"   Need: {0.60 - best_f1:+.4f} improvement")
    
    print("=" * 60)
    
    return results

def analyze_results():
    """Analyze training results"""
    results_file = './experiments/results/phase2c/quick_training/results.json'
    
    if not os.path.exists(results_file):
        print("No results found. Run training first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    print(f"Best F1: {results['best_f1']:.4f}")
    print(f"Phase 2B baseline: {results['phase2b_baseline']:.4f}")
    print(f"Improvement: {results['improvement']:+.4f}")
    print(f"Target achieved: {'✅ YES' if results['achieved_target'] else '❌ NO'}")
    
    if results['best_f1'] > 0.60:
        print("\n🎉 PHASE 2C COMPLETE!")
        print("Ready for paper preparation.")
        
        # Create simple comparison table
        print("\nPerformance Comparison:")
        print("=" * 40)
        print(f"{'Phase':<20} {'F1 Score':<10}")
        print("-" * 40)
        print(f"{'1. Baseline (Geometric)':<20} {'0.077':<10}")
        print(f"{'2. Phase 2A (Base GNN)':<20} {'0.3077':<10}")
        print(f"{'3. Phase 2B (HamGNN 70)':<20} {'0.5444':<10}")
        print(f"{'4. Phase 2C (Final)':<20} {results['best_f1']:.4f}")
        print("=" * 40)
        
        # Calculate total improvement
        total_improvement = results['best_f1'] - 0.077
        print(f"\nTotal improvement from baseline: {total_improvement:.4f}")
        print(f"Relative improvement: {total_improvement/0.077*100:.1f}%")
        
    else:
        print("\n❌ Target not achieved.")
        print("Consider:")
        print("  1. Training longer (more epochs)")
        print("  2. Tuning hyperparameters further")
        print("  3. Adding more features")
        print("  4. Scaling to full dataset")

if __name__ == "__main__":
    # Run quick training
    results = run_quick_training()
    
    # Analyze results
    if results:
        analyze_results()
