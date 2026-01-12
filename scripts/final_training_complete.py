#!/usr/bin/env python3
"""
PHASE 2C FINAL: Train on full dataset, evaluate on test set
"""
import os
import torch
import json
import numpy as np
import time
from torch_geometric.loader import DataLoader

sys.path.append('./src/models')
sys.path.append('./src/training')
from hamiltonian_gnn_simple import HamiltonianGNNsimple
from physics_trainer import PhysicsTrainer

def load_full_data():
    """Load all processed data"""
    print("Loading full dataset...")
    
    splits = {}
    for split in ['train', 'val', 'test']:
        split_dir = f'./data/processed/physics_graphs_full/{split}/'
        if not os.path.exists(split_dir):
            print(f"ERROR: {split_dir} not found")
            return None, None, None
        
        files = [f for f in os.listdir(split_dir) if f.endswith('.pt')]
        graphs = []
        for f in files[:20] if split == 'test' else files:  # Load all train/val, sample test
            graphs.append(torch.load(os.path.join(split_dir, f)))
        
        splits[split] = graphs
        print(f"  {split}: {len(graphs)} graphs")
    
    return splits['train'], splits['val'], splits['test']

def final_training():
    """Final training to achieve F1 > 0.60"""
    print("=" * 70)
    print("PHASE 2C FINAL: TRAINING ON FULL DATASET")
    print("Target: F1 > 0.60 on test set")
    print("=" * 70)
    
    # Load data
    train_graphs, val_graphs, test_graphs = load_full_data()
    if not train_graphs or len(train_graphs) < 500:
        print("ERROR: Not enough training data")
        return None
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_graphs)} proteins")
    print(f"  Validation: {len(val_graphs)} proteins")
    print(f"  Test: {len(test_graphs)} proteins (held-out)")
    
    # Load optimized hyperparameters
    with open('./experiments/results/phase2c/hyperparameter_optimization/best_params.json', 'r') as f:
        best_params = json.load(f)
    
    hyperparams = best_params['best_params']
    print(f"\nUsing optimized hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create model
    input_dim = train_graphs[0].x.shape[1]
    model = HamiltonianGNNsimple(
        input_dim=input_dim,
        hidden_dim=hyperparams.get('hidden_dim', 128),
        lambda_physics=hyperparams.get('lambda_physics', 0.0001),
        dropout=hyperparams.get('dropout', 0.3)
    )
    
    # Trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams.get('learning_rate', 0.001),
        pos_weight=hyperparams.get('pos_weight', 20.0),
        lambda_physics=hyperparams.get('lambda_physics', 0.0001)
    )
    
    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=hyperparams.get('batch_size', 8), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=hyperparams.get('batch_size', 8), shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=hyperparams.get('batch_size', 8), shuffle=False)
    
    # TRAIN
    print("\n" + "=" * 70)
    print("STARTING FINAL TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=200,
        early_stopping_patience=30,
        save_dir='./experiments/results/phase2c/final_model',
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating on held-out test set...")
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Results
    results = {
        'test_performance': {
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'auc': float(test_auc),
            'loss': float(test_loss)
        },
        'training_info': {
            'best_val_f1': float(max(history['val_f1']) if history['val_f1'] else 0),
            'training_time_hours': float(training_time / 3600),
            'epochs_trained': len(history['train_loss']),
            'physics_ratio': float(history['physics_loss_ratio'][-1] if history['physics_loss_ratio'] else 0)
        },
        'hyperparameters': hyperparams,
        'dataset_sizes': {
            'train': len(train_graphs),
            'val': len(val_graphs),
            'test': len(test_graphs)
        },
        'comparison': {
            'phase1_baseline': 0.077,
            'phase2a_base_gnn': 0.3077,
            'phase2b_hamgnn_70': 0.5444,
            'phase2c_final': float(test_f1),
            'improvement_over_phase2b': float(test_f1 - 0.5444),
            'improvement_over_baseline': float(test_f1 - 0.077),
            'beats_fpocket': test_f1 > 0.52
        }
    }
    
    # Save
    output_dir = './experiments/results/phase2c/final_results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/final_performance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), f'{output_dir}/final_model.pt')
    
    # Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test Set F1: {test_f1:.4f}")
    print(f"Target (F1 > 0.60): {'✅ ACHIEVED' if test_f1 > 0.60 else '❌ NOT ACHIEVED'}")
    print(f"Training time: {training_time/3600:.1f} hours")
    print(f"Improvement over Phase 2B: {test_f1 - 0.5444:+.4f}")
    print(f"Beats FPOCKET (0.52): {'✅ YES' if test_f1 > 0.52 else '❌ NO'}")
    print("=" * 70)
    
    if test_f1 > 0.60:
        print("\n🎉🎉🎉 PHASE 2C COMPLETE AND SUCCESSFUL! 🎉🎉🎉")
        with open('./experiments/results/phase2c/PHASE2C_OFFICIAL_SUCCESS.txt', 'w') as f:
            f.write(f"PHASE 2C OFFICIALLY COMPLETE\n")
            f.write(f"Test F1: {test_f1:.4f} > 0.60 target\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        print(f"\n⚠️ Phase 2C target not achieved. F1 = {test_f1:.4f}")
    
    return results

if __name__ == "__main__":
    final_training()
