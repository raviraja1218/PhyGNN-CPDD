#!/usr/bin/env python3
"""
PHASE 2C EMERGENCY FINAL PUSH - FIXED
"""
import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
import time

sys.path.append('./src/models')
sys.path.append('./src/training')

from hamiltonian_gnn_simple import HamiltonianGNNsimple
from physics_trainer import PhysicsTrainer

def load_emergency_graphs():
    """Load emergency processed graphs"""
    print("Loading emergency processed graphs...")
    
    graph_dir = './data/processed/emergency_200/'
    if not os.path.exists(graph_dir):
        print(f"ERROR: Emergency graphs not found at {graph_dir}")
        print("Run process_200_emergency_fixed.py first!")
        return None, None, None  # FIXED: Return 3 values
    
    files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    print(f"Found {len(files)} emergency graphs")
    
    graphs = []
    for f in tqdm(files[:150], desc="Loading"):  # Use first 150 for speed
        try:
            graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    print(f"Loaded {len(graphs)} graphs")
    
    if len(graphs) < 30:
        print(f"ERROR: Not enough graphs loaded ({len(graphs)} < 30)")
        return None, None, None
    
    # Split: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]
    
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs  # FIXED: Return 3 values

def emergency_final_training():
    """Final emergency training to reach F1 > 0.60"""
    print("=" * 70)
    print("PHASE 2C EMERGENCY FINAL TRAINING")
    print("GOAL: Achieve F1 > 0.60 with 200 proteins")
    print("=" * 70)
    
    # Load graphs
    train_graphs, val_graphs, test_graphs = load_emergency_graphs()
    if train_graphs is None or len(train_graphs) < 50:
        print("ERROR: Not enough graphs loaded")
        return None
    
    # Check feature dimension
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # ULTRA-OPTIMIZED Hyperparameters
    hyperparams = {
        'lambda_physics': 0.00001,  # MINIMAL physics
        'learning_rate': 0.0003,
        'batch_size': 8,
        'hidden_dim': 128,
        'pos_weight': 10.0,  # Less aggressive
        'dropout': 0.2,
        'epochs': 200,
        'patience': 40
    }
    
    print("\nULTRA-OPTIMIZED Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create model (SIMPLIFIED, minimal physics)
    model = HamiltonianGNNsimple(
        input_dim=input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        lambda_physics=hyperparams['lambda_physics'],
        dropout=hyperparams['dropout']
    )
    
    # Create trainer with VERY low physics weight
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams['learning_rate'],
        pos_weight=hyperparams['pos_weight'],
        lambda_physics=hyperparams['lambda_physics']
    )
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    
    # TRAIN!
    print("\n" + "=" * 70)
    print("STARTING FINAL EMERGENCY TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hyperparams['epochs'],
        early_stopping_patience=hyperparams['patience'],
        save_dir='./experiments/results/phase2c/emergency_final',
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Get best validation F1
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    # Test on held-out test set
    print("\nEvaluating on held-out test set...")
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Results
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'best_val_f1': float(best_val_f1),
        'phase2b_baseline': 0.5444,
        'improvement_over_phase2b': float(test_f1 - 0.5444),
        'hyperparameters': hyperparams,
        'training_time_minutes': float(training_time / 60),
        'num_train_graphs': len(train_graphs),
        'num_val_graphs': len(val_graphs),
        'num_test_graphs': len(test_graphs),
        'achieved_target': bool(test_f1 > 0.60),
        'physics_ratio_at_best': float(history['physics_loss_ratio'][np.argmax(history['val_f1'])] if history['val_f1'] else 0)
    }
    
    # Save results
    output_dir = './experiments/results/phase2c/emergency_final'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), f'{output_dir}/final_model.pt')
    
    # Save history
    with open(f'{output_dir}/training_history.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        json.dump(convert(history), f, indent=2)
    
    # PRINT FINAL RESULTS
    print("\n" + "=" * 70)
    print("FINAL EMERGENCY TRAINING RESULTS")
    print("=" * 70)
    print(f"Test Set F1:       {test_f1:.4f}")
    print(f"Test Set Precision: {test_precision:.4f}")
    print(f"Test Set Recall:    {test_recall:.4f}")
    print(f"Test Set AUC:       {test_auc:.4f}")
    print(f"Best Val F1:        {best_val_f1:.4f}")
    print(f"Training time:      {training_time/60:.1f} minutes")
    print(f"Physics ratio:      {results['physics_ratio_at_best']:.3f}")
    print(f"Target (F1 > 0.60): {'✅ ACHIEVED' if test_f1 > 0.60 else '❌ NOT ACHIEVED'}")
    print("=" * 70)
    
    if test_f1 > 0.60:
        print("\n🎉🎉🎉 PHASE 2C COMPLETE AND SUCCESSFUL! 🎉🎉🎉")
        print(f"   Achieved F1 = {test_f1:.4f} (target: >0.60)")
        
        # Create success flag
        with open('./experiments/results/phase2c/PHASE2C_SUCCESS.txt', 'w') as f:
            f.write(f"PHASE 2C COMPLETE\n")
            f.write(f"Test F1: {test_f1:.4f}\n")
            f.write(f"Target: >0.60\n")
            f.write(f"Status: SUCCESS\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create performance comparison
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON ACROSS ALL PHASES")
        print("=" * 70)
        print(f"{'Phase':<25} {'F1 Score':<10} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'1. Baseline (Geometric)':<25} {'0.077':<10} {'-':<15}")
        print(f"{'2. Phase 2A (Base GNN)':<25} {'0.3077':<10} {'+300%':<15}")
        print(f"{'3. Phase 2B (HamGNN 70)':<25} {'0.5444':<10} {'+76.9%':<15}")
        
        improvement_pct = (test_f1 - 0.5444) / 0.5444 * 100
        print(f"{'4. Phase 2C (Final 200)':<25} {test_f1:.4f} {'+'+str(round(improvement_pct, 1))+'%':<15}")
        print("=" * 70)
        
        total_improvement = test_f1 - 0.077
        print(f"\nTotal improvement from baseline: {total_improvement:.4f}")
        print(f"Relative improvement: {total_improvement/0.077*100:.1f}%")
        
        # Compare with FPOCKET
        print(f"\nComparison with State-of-the-Art:")
        print(f"  Our method (Phase 2C): F1 = {test_f1:.4f}")
        print(f"  FPOCKET (literature):  F1 = 0.5200")
        print(f"  Advantage:              +{test_f1 - 0.52:.4f}")
        
    else:
        print(f"\n❌ Target not achieved. F1 = {test_f1:.4f}")
        print(f"   Need: {0.60 - test_f1:+.4f} improvement")
    
    return results

if __name__ == "__main__":
    results = emergency_final_training()
    
    if results and results['achieved_target']:
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR PAPER:")
        print("1. Results saved to: ./experiments/results/phase2c/emergency_final/")
        print("2. Model saved as: final_model.pt")
        print("3. Create paper figures")
        print("4. Write manuscript")
        print("=" * 70)
