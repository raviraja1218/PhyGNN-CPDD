#!/usr/bin/env python3
"""
Final training on full dataset with optimized hyperparameters
Evaluate on held-out test set
"""
import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import time

# Add src to path
sys.path.append('./src/models')
from hamiltonian_gnn_phase2c import HamiltonianGNN
from physics_trainer import PhysicsTrainer

def load_all_graphs():
    """Load all graphs from processed directories"""
    print("Loading all graphs...")
    
    graphs = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = f"./data/processed/physics_graphs/full/{split}"
        split_graphs = []
        
        print(f"  Loading {split} graphs...")
        graph_files = os.listdir(split_dir)
        
        for gf in tqdm(graph_files, desc=f"  {split}"):
            if gf.endswith('.pt'):
                try:
                    graph = torch.load(os.path.join(split_dir, gf))
                    split_graphs.append(graph)
                except Exception as e:
                    print(f"    Warning: Could not load {gf}: {e}")
        
        graphs[split] = split_graphs
        print(f"    Loaded {len(split_graphs)} {split} graphs")
    
    return graphs

def train_final_model(hyperparams):
    """Train final model with optimized hyperparameters"""
    print("=" * 60)
    print("FINAL TRAINING ON FULL DATASET")
    print("=" * 60)
    
    # Load all graphs
    graphs = load_all_graphs()
    
    # Create data loaders
    train_loader = DataLoader(graphs['train'], 
                             batch_size=hyperparams['batch_size'],
                             shuffle=True)
    val_loader = DataLoader(graphs['val'],
                           batch_size=hyperparams['batch_size'],
                           shuffle=False)
    test_loader = DataLoader(graphs['test'],
                            batch_size=hyperparams['batch_size'],
                            shuffle=False)
    
    print(f"Training on {len(graphs['train'])} proteins")
    print(f"Validating on {len(graphs['val'])} proteins")
    print(f"Testing on {len(graphs['test'])} proteins (held-out)")
    
    # Create model
    input_dim = graphs['train'][0].x.shape[1]
    model = HamiltonianGNN(
        input_dim=input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        lambda_physics=hyperparams['lambda_physics']
    )
    
    print(f"Model input dimension: {input_dim}")
    print(f"Model hidden dimension: {hyperparams['hidden_dim']}")
    print(f"Physics weight λ: {hyperparams['lambda_physics']}")
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=hyperparams['learning_rate'],
        pos_weight=20.0  # 1/0.046 ≈ 21.7, using 20
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=20,
        save_dir="./experiments/results/phase2c/week3/final_model/model_checkpoints"
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating on held-out test set...")
    test_loss, test_f1, test_precision, test_recall, test_auc = trainer.evaluate(test_loader)
    
    # Save final results
    results = {
        'hyperparameters': hyperparams,
        'test_performance': {
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'auc': float(test_auc),
            'loss': float(test_loss)
        },
        'training_history': {
            'best_val_f1': float(max(history['val_f1'])),
            'final_val_f1': float(history['val_f1'][-1]),
            'final_train_loss': float(history['train_loss'][-1]),
            'training_time_seconds': float(training_time),
            'epochs_trained': len(history['train_loss'])
        },
        'comparison_with_baselines': {
            'phase1_baseline_f1': 0.077,
            'phase2a_base_gnn_f1': 0.3077,
            'phase2b_hamgnn_70proteins_f1': 0.5444,
            'phase2c_final_hamgnn_f1': float(test_f1),
            'improvement_over_phase2b': float(test_f1 - 0.5444),
            'improvement_over_baseline': float(test_f1 - 0.077)
        },
        'dataset_statistics': {
            'train_proteins': len(graphs['train']),
            'val_proteins': len(graphs['val']),
            'test_proteins': len(graphs['test']),
            'total_proteins': len(graphs['train']) + len(graphs['val']) + len(graphs['test'])
        }
    }
    
    # Save results
    output_dir = "./experiments/results/phase2c/week3/final_model"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/final_performance.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
    
    # Save test predictions for analysis
    save_test_predictions(trainer, test_loader, graphs['test'], output_dir)
    
    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print(f"Test Set F1: {test_f1:.4f}")
    print(f"Test Set Precision: {test_precision:.4f}")
    print(f"Test Set Recall: {test_recall:.4f}")
    print(f"Test Set AUC: {test_auc:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print("=" * 60)
    
    if test_f1 > 0.60:
        print("🎉 SUCCESS: Exceeded target F1 > 0.60!")
    else:
        print("⚠️ Warning: Below target F1 > 0.60")
        print(f"   Current: {test_f1:.4f}, Target: 0.60")
    
    return results

def save_test_predictions(trainer, test_loader, test_graphs, output_dir):
    """Save detailed test predictions for analysis"""
    trainer.model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(trainer.device)
            logits = trainer.model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Get protein IDs
            batch_protein_ids = batch.protein_id if hasattr(batch, 'protein_id') else [f"batch_{i}"]
            
            for j in range(batch.num_graphs if hasattr(batch, 'num_graphs') else 1):
                protein_id = batch_protein_ids[j] if isinstance(batch_protein_ids, list) else batch_protein_ids
                
                # Get slice for this graph
                start_idx = batch.ptr[j] if hasattr(batch, 'ptr') else 0
                end_idx = batch.ptr[j+1] if hasattr(batch, 'ptr') else len(preds)
                
                graph_preds = preds[start_idx:end_idx].cpu().numpy().flatten()
                graph_probs = probs[start_idx:end_idx].cpu().numpy().flatten()
                graph_labels = batch.y[start_idx:end_idx].cpu().numpy().flatten()
                
                predictions.append({
                    'protein_id': protein_id,
                    'predictions': graph_preds.tolist(),
                    'probabilities': graph_probs.tolist(),
                    'labels': graph_labels.tolist(),
                    'num_nodes': len(graph_preds),
                    'num_pocket_predicted': int(np.sum(graph_preds)),
                    'num_pocket_actual': int(np.sum(graph_labels))
                })
    
    # Save predictions
    import pandas as pd
    df = pd.DataFrame(predictions)
    df.to_csv(f"{output_dir}/test_set_predictions.csv", index=False)
    
    print(f"Saved {len(predictions)} protein predictions to CSV")

def create_performance_plots(results):
    """Create performance visualization plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir = "./experiments/results/phase2c/week3/final_model/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Baseline comparison bar chart
    plt.figure(figsize=(10, 6))
    baselines = ['Phase 1\n(Geometric)', 'Phase 2A\n(Base GNN)', 'Phase 2B\n(HamGNN 70)', 'Phase 2C\n(Final HamGNN)']
    f1_scores = [0.077, 0.3077, 0.5444, results['test_performance']['f1']]
    
    bars = plt.bar(baselines, f1_scores, color=['lightgray', 'lightblue', 'orange', 'green'])
    plt.ylabel('F1 Score')
    plt.title('Performance Improvement Across Phases')
    plt.ylim(0, 0.7)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{output_dir}/baseline_comparison.png", dpi=300, bbox_inches='tight')
    
    # 2. Performance metrics radar chart
    metrics = ['Precision', 'Recall', 'F1', 'AUC']
    values = [
        results['test_performance']['precision'],
        results['test_performance']['recall'],
        results['test_performance']['f1'],
        results['test_performance']['auc']
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Close the radar chart
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title('Test Set Performance Metrics')
    plt.savefig(f"{output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # Load optimal hyperparameters
    with open("./experiments/results/phase2c/week2/hyperparameter_optimization/optimal_parameters.json", 'r') as f:
        optimal_data = json.load(f)
    
    hyperparams = optimal_data['best_params']
    
    print(f"Using optimal hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Train final model
    results = train_final_model(hyperparams)
    
    # Create performance plots
    create_performance_plots(results)
