#!/usr/bin/env python3
"""
Hyperparameter optimization for Hamiltonian GNN
Focus: λ (physics weight), learning rate, batch size
"""
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
import itertools
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.append('./src/models')
from hamiltonian_gnn_phase2c import HamiltonianGNN
from physics_trainer import PhysicsTrainer

def load_small_dataset(max_proteins=50):
    """Load small subset for quick hyperparameter search"""
    print("Loading small dataset for hyperparameter search...")
    
    graphs = []
    train_dir = "./data/processed/physics_graphs/full/train"
    protein_files = os.listdir(train_dir)[:max_proteins]
    
    for pf in tqdm(protein_files, desc="Loading graphs"):
        graph_path = os.path.join(train_dir, pf)
        try:
            graph = torch.load(graph_path)
            graphs.append(graph)
        except:
            continue
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs

def hyperparameter_grid_search():
    """Perform grid search over hyperparameters"""
    # Load small dataset
    graphs = load_small_dataset(max_proteins=50)
    
    if len(graphs) < 10:
        print("Error: Need at least 10 graphs for search")
        return
    
    # Split into train/val
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    # Hyperparameter grid
    param_grid = {
        'lambda_physics': [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],  # λ values
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
        'batch_size': [4, 8, 16],
        'hidden_dim': [64, 128, 256]
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations")
    print("Expected time: ~2-3 hours")
    
    results = []
    best_f1 = 0
    best_params = None
    
    # Test each combination
    for i, params in enumerate(tqdm(param_combinations, desc="Grid search")):
        param_dict = dict(zip(param_names, params))
        
        try:
            # Create model
            input_dim = train_graphs[0].x.shape[1]
            model = HamiltonianGNN(
                input_dim=input_dim,
                hidden_dim=param_dict['hidden_dim'],
                lambda_physics=param_dict['lambda_physics']
            )
            
            # Create data loaders
            train_loader = DataLoader(train_graphs, 
                                     batch_size=param_dict['batch_size'],
                                     shuffle=True)
            val_loader = DataLoader(val_graphs,
                                   batch_size=param_dict['batch_size'],
                                   shuffle=False)
            
            # Create trainer
            trainer = PhysicsTrainer(
                model=model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                learning_rate=param_dict['learning_rate'],
                pos_weight=20.0  # Class imbalance (4.6% positive)
            )
            
            # Train for 5 epochs (quick evaluation)
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                save_dir=None,  # Don't save models during search
                verbose=False
            )
            
            # Get best validation F1
            best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
            
            # Store results
            result = {
                'params': param_dict,
                'best_val_f1': best_val_f1,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                'physics_contribution': history.get('physics_loss_ratio', [0])[-1]
            }
            
            results.append(result)
            
            # Update best
            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                best_params = param_dict
                print(f"\nNew best: F1={best_val_f1:.4f} with {param_dict}")
            
        except Exception as e:
            print(f"Error with params {param_dict}: {e}")
            continue
    
    # Save results
    results.sort(key=lambda x: x['best_val_f1'], reverse=True)
    
    output_dir = "./experiments/results/phase2c/week2/hyperparameter_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all results
    with open(f"{output_dir}/grid_search_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save top 10
    top_results = results[:10]
    with open(f"{output_dir}/top_10_results.json", 'w') as f:
        json.dump(top_results, f, indent=2)
    
    # Save best parameters
    best_output = {
        'best_f1': best_f1,
        'best_params': best_params,
        'phase2b_baseline_f1': 0.5444,
        'improvement': best_f1 - 0.5444
    }
    
    with open(f"{output_dir}/optimal_parameters.json", 'w') as f:
        json.dump(best_output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Improvement over Phase 2B: {best_f1 - 0.5444:.4f}")
    print("=" * 60)
    
    return best_params

def create_analysis_plots():
    """Create plots from hyperparameter search results"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load results
    with open("./experiments/results/phase2c/week2/hyperparameter_optimization/grid_search_results.json", 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df_data = []
    for r in results:
        row = r['params'].copy()
        row['f1'] = r['best_val_f1']
        row['physics_ratio'] = r.get('physics_contribution', 0)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Create output directory
    output_dir = "./experiments/results/phase2c/week2/hyperparameter_optimization/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. λ vs F1 plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lambda_physics'], df['f1'], alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Physics weight (λ)')
    plt.ylabel('Validation F1')
    plt.title('Physics Weight vs Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lambda_vs_f1.png", dpi=300, bbox_inches='tight')
    
    # 2. Learning rate vs F1
    plt.figure(figsize=(10, 6))
    plt.scatter(df['learning_rate'], df['f1'], alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation F1')
    plt.title('Learning Rate vs Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lr_vs_f1.png", dpi=300, bbox_inches='tight')
    
    # 3. 3D scatter: λ vs LR vs F1
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(np.log10(df['lambda_physics']), 
                        np.log10(df['learning_rate']), 
                        df['f1'],
                        c=df['f1'], cmap='viridis')
    ax.set_xlabel('log10(λ)')
    ax.set_ylabel('log10(Learning Rate)')
    ax.set_zlabel('F1 Score')
    ax.set_title('Hyperparameter Optimization Space')
    plt.colorbar(scatter)
    plt.savefig(f"{output_dir}/3d_hyperparameter_space.png", dpi=300, bbox_inches='tight')
    
    # 4. Pareto frontier: F1 vs Physics Contribution
    plt.figure(figsize=(10, 6))
    plt.scatter(df['physics_ratio'], df['f1'], alpha=0.6)
    plt.xlabel('Physics Contribution Ratio')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Physics Balance')
    plt.grid(True, alpha=0.3)
    
    # Highlight Pareto frontier
    frontier = []
    sorted_by_physics = df.sort_values('physics_ratio')
    current_max_f1 = -1
    for _, row in sorted_by_physics.iterrows():
        if row['f1'] > current_max_f1:
            frontier.append((row['physics_ratio'], row['f1']))
            current_max_f1 = row['f1']
    
    frontier_x, frontier_y = zip(*frontier)
    plt.plot(frontier_x, frontier_y, 'r--', linewidth=2, label='Pareto Frontier')
    plt.legend()
    plt.savefig(f"{output_dir}/pareto_frontier.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # Run hyperparameter search
    best_params = hyperparameter_grid_search()
    
    # Create analysis plots
    create_analysis_plots()
