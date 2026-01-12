#!/usr/bin/env python3
"""
Final training for Phase 2C completion
Train on 240, validate on 30, test on 30 physics-enhanced proteins
"""
import os
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
import time

sys.path.append('./src/models')
sys.path.append('./src/training')
from hamgnn_final import HamiltonianGNN
from trainer_final import PhysicsTrainer

def load_final_dataset():
    """Load the 300 final graphs"""
    graphs = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = f"./data/processed/phase2c_final_300/{split}"
        split_graphs = []
        
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.endswith('.pt'):
                    try:
                        g = torch.load(os.path.join(split_dir, f))
                        split_graphs.append(g)
                    except:
                        continue
        
        graphs[split] = split_graphs
        print(f"Loaded {len(split_graphs)} {split} graphs")
    
    return graphs

def train_final_model():
    """Train final model with optimal hyperparameters"""
    print("=" * 60)
    print("PHASE 2C FINAL TRAINING")
    print("=" * 60)
    
    # Load graphs
    graphs = load_final_dataset()
    
    if len(graphs['train']) < 100:
        print("ERROR: Need at least 100 training graphs")
        return
    
    # Create data loaders
    train_loader = DataLoader(graphs['train'], batch_size=8, shuffle=True)
    val_loader = DataLoader(graphs['val'], batch_size=8, shuffle=False)
    test_loader = DataLoader(graphs['test'], batch_size=8, shuffle=False)
    
    print(f"Training: {len(graphs['train'])} graphs")
    print(f"Validation: {len(graphs['val'])} graphs")
    print(f"Testing: {len(graphs['test'])} graphs")
    
    # Create model with optimal parameters from Phase 2B
    input_dim = graphs['train'][0].x.shape[1]
    model = HamiltonianGNN(
        input_dim=input_dim,
        hidden_dim=128,
        lambda_physics=0.0001  # Optimal from Phase 2B
    )
    
    print(f"Model: {input_dim} input features, λ=0.0001")
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        pos_weight=20.0
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=10,
        save_dir="./experiments/results/phase2c_final/models"
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        'test_performance': {
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'auc': float(test_auc),
            'loss': float(test_loss)
        },
        'training_info': {
            'train_graphs': len(graphs['train']),
            'val_graphs': len(graphs['val']),
            'test_graphs': len(graphs['test']),
            'training_time_minutes': train_time / 60,
            'best_val_f1': float(max(history['val_f1'])),
            'final_val_f1': float(history['val_f1'][-1])
        },
        'hyperparameters': {
            'lambda_physics': 0.0001,
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'batch_size': 8,
            'pos_weight': 20.0
        },
        'comparison': {
            'phase1_baseline': 0.077,
            'phase2a_base_gnn': 0.3077,
            'phase2b_70proteins': 0.5444,
            'phase2c_300proteins': float(test_f1)
        }
    }
    
    # Save
    os.makedirs("./experiments/results/phase2c_final", exist_ok=True)
    with open("./experiments/results/phase2c_final/final_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), "./experiments/results/phase2c_final/final_model.pt")
    
    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Training time: {train_time/60:.1f} minutes")
    
    if test_f1 > 0.50:
        print("✅ SUCCESS: Achieved target F1 > 0.50!")
    else:
        print("⚠️ Below target but still competitive")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    train_final_model()
