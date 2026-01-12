#!/usr/bin/env python3
"""
PHASE 2C FINAL PROPER TRAINING
Using Phase 2B proven recipe with 200 proteins
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

# Import the CORRECT model that matches Phase 2B
try:
    # Try to import the actual Phase 2B model
    from improved_gnn_fixed import ImprovedGNN as ModelClass
    print("Using ImprovedGNN (Phase 2B model)")
except:
    # Fallback to HamiltonianGNNsimple
    from hamiltonian_gnn_simple import HamiltonianGNNsimple as ModelClass
    print("Using HamiltonianGNNsimple")

from physics_trainer import PhysicsTrainer

def load_all_graphs():
    """Load all emergency processed graphs"""
    print("Loading all emergency graphs...")
    
    graph_dir = './data/processed/emergency_200/'
    files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    
    graphs = []
    for f in tqdm(files, desc="Loading"):
        try:
            graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    print(f"Loaded {len(graphs)} graphs")
    
    # More aggressive split: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(graphs))
    val_size = int(0.15 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]
    
    print(f"  Train: {len(train_graphs)} graphs (70%)")
    print(f"  Val: {len(val_graphs)} graphs (15%)")
    print(f"  Test: {len(test_graphs)} graphs (15%)")
    
    return train_graphs, val_graphs, test_graphs

def analyze_class_balance(graphs):
    """Analyze class balance in the graphs"""
    total_nodes = 0
    total_positives = 0
    
    for graph in graphs:
        if hasattr(graph, 'y'):
            total_nodes += graph.y.shape[0]
            total_positives += graph.y.sum().item()
    
    if total_nodes > 0:
        positive_ratio = total_positives / total_nodes
        print(f"Class balance: {positive_ratio:.3%} positive ({total_positives}/{total_nodes})")
        return positive_ratio
    return 0.046  # Default from Phase 2B

def train_with_phase2b_recipe():
    """Train using Phase 2B proven recipe"""
    print("=" * 70)
    print("PHASE 2C: USING PHASE 2B PROVEN RECIPE")
    print("=" * 70)
    
    # Load graphs
    train_graphs, val_graphs, test_graphs = load_all_graphs()
    if len(train_graphs) < 50:
        print("ERROR: Not enough graphs")
        return None
    
    # Analyze class balance
    pos_ratio = analyze_class_balance(train_graphs)
    pos_weight = 1.0 / pos_ratio if pos_ratio > 0 else 20.0
    print(f"Using pos_weight: {pos_weight:.1f}")
    
    # PHASE 2B PROVEN HYPERPARAMETERS
    hyperparams = {
        'lambda_physics': 0.0001,  # PHASE 2B OPTIMAL
        'learning_rate': 0.001,    # PHASE 2B OPTIMAL
        'batch_size': 8,           # PHASE 2B OPTIMAL
        'hidden_dim': 128,         # PHASE 2B OPTIMAL
        'pos_weight': pos_weight,
        'dropout': 0.3,            # PHASE 2B OPTIMAL
        'epochs': 150,
        'patience': 25
    }
    
    print("\nUSING PHASE 2B PROVEN HYPERPARAMETERS:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create model
    input_dim = train_graphs[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Try to match Phase 2B architecture
    if input_dim == 30:  # Same as Phase 2B
        model = ModelClass(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            lambda_physics=hyperparams['lambda_physics'],
            dropout=hyperparams['dropout']
        )
    else:
        # Adjust if different feature dimension
        model = ModelClass(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
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
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=hyperparams['batch_size'], shuffle=False)
    
    # TRAIN
    print("\n" + "=" * 70)
    print("STARTING TRAINING WITH PHASE 2B RECIPE")
    print("=" * 70)
    
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hyperparams['epochs'],
        early_stopping_patience=hyperparams['patience'],
        save_dir='./experiments/results/phase2c/final_proper',
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Get best validation F1
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    # Test on held-out test set
    print("\nEvaluating on held-out test set...")
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Calculate additional metrics
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    # Get predictions for confusion matrix
    trainer.model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(trainer.device)
            logits, _ = trainer.model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Results
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'best_val_f1': float(best_val_f1),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'phase2b_baseline': 0.5444,
        'improvement_over_phase2b': float(test_f1 - 0.5444),
        'hyperparameters': hyperparams,
        'training_time_minutes': float(training_time / 60),
        'num_train_graphs': len(train_graphs),
        'num_val_graphs': len(val_graphs),
        'num_test_graphs': len(test_graphs),
        'class_balance': float(pos_ratio),
        'achieved_target': bool(test_f1 > 0.60),
        'physics_ratio_at_best': float(history['physics_loss_ratio'][np.argmax(history['val_f1'])] if history['val_f1'] else 0)
    }
    
    # Save results
    output_dir = './experiments/results/phase2c/final_proper'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), f'{output_dir}/final_model.pt')
    
    # Save training history
    def convert_numpy(obj):
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
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(convert_numpy(history), f, indent=2)
    
    # PRINT RESULTS
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test Set F1:       {test_f1:.4f}")
    print(f"Test Set Precision: {test_precision:.4f}")
    print(f"Test Set Recall:    {test_recall:.4f}")
    print(f"Test Set AUC:       {test_auc:.4f}")
    print(f"Best Val F1:        {best_val_f1:.4f}")
    print(f"Training time:      {training_time/60:.1f} minutes")
    print(f"Physics ratio:      {results['physics_ratio_at_best']:.3f}")
    print(f"Phase 2B baseline:  0.5444")
    print(f"Improvement:        {test_f1 - 0.5444:+.4f}")
    print(f"Target (F1 > 0.60): {'✅ ACHIEVED' if test_f1 > 0.60 else '❌ NOT ACHIEVED'}")
    print("=" * 70)
    
    if test_f1 > 0.60:
        print("\n🎉🎉🎉 PHASE 2C COMPLETE AND SUCCESSFUL! 🎉🎉🎉")
        
        # Create success flag
        with open('./experiments/results/phase2c/PHASE2C_SUCCESS.txt', 'w') as f:
            f.write(f"PHASE 2C COMPLETE\n")
            f.write(f"Test F1: {test_f1:.4f}\n")
            f.write(f"Target: >0.60\n")
            f.write(f"Status: SUCCESS\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: Phase 2B recipe with 200 proteins\n")
        
        # Performance comparison
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Phase':<25} {'F1 Score':<10} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'1. Baseline':<25} {'0.077':<10} {'-':<15}")
        print(f"{'2. Phase 2A':<25} {'0.3077':<10} {'+300%':<15}")
        print(f"{'3. Phase 2B (70)':<25} {'0.5444':<10} {'+76.9%':<15}")
        
        improvement_pct = (test_f1 - 0.5444) / 0.5444 * 100
        print(f"{'4. Phase 2C (200)':<25} {test_f1:.4f} {'+'+str(round(improvement_pct, 1))+'%':<15}")
        print("=" * 70)
        
        total_improvement = test_f1 - 0.077
        print(f"\nTotal improvement: {total_improvement:.4f} ({total_improvement/0.077*100:.1f}%)")
        print(f"Beats FPOCKET (0.52): +{test_f1 - 0.52:.4f}")
        
    else:
        print(f"\n❌ Target not achieved: {test_f1:.4f} < 0.60")
        print(f"   But Phase 2B already achieved 0.5444 (beats FPOCKET)")
    
    return results

if __name__ == "__main__":
    results = train_with_phase2b_recipe()
    
    if results and results['achieved_target']:
        print("\n" + "=" * 70)
        print("READY FOR PAPER SUBMISSION!")
        print("=" * 70)
        print("1. All results saved")
        print("2. Model trained and tested")
        print("3. Performance documented")
        print("4. Beats state-of-the-art")
        print("=" * 70)
