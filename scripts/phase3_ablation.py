#!/usr/bin/env python3
"""
Ablation Study: Test which physics components matter most
"""
import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.append('./src/models')
from hamiltonian_gnn_ablation_fixed import HamiltonianGNN
from physics_trainer import PhysicsTrainer

def load_subset_graphs(num_proteins=30):
    """Load subset of graphs for quick ablation study"""
    print(f"Loading {num_proteins} graphs for ablation study...")
    
    graphs = []
    data_dir = "./data/processed/physics_graphs/train"
    graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')][:num_proteins]
    
    for gf in tqdm(graph_files, desc="Loading graphs"):
        try:
            graph = torch.load(os.path.join(data_dir, gf))
            graphs.append(graph)
        except Exception as e:
            print(f"  Warning: Could not load {gf}: {e}")
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs

def train_and_evaluate(model_config, graphs, name="baseline"):
    """Train and evaluate a specific model configuration"""
    # Split graphs into train/val
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Create model with specific physics components disabled
    input_dim = train_graphs[0].x.shape[1]
    model = HamiltonianGNN(
        input_dim=input_dim,
        hidden_dim=128,
        lambda_physics=0.0001,
        disable_components=model_config.get('disable_components', [])
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        pos_weight=20.0
    )
    
    # Train for 20 epochs (quick evaluation)
    print(f"\nTraining {name}...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        save_dir=None,
        verbose=False
    )
    
    # Get best validation F1
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
    
    return {
        'name': name,
        'best_f1': float(best_f1),
        'final_train_loss': float(final_train_loss),
        'physics_ratio': history.get('physics_loss_ratio', [0])[-1],
        'disable_components': model_config.get('disable_components', [])
    }

def run_ablation_study():
    """Run complete ablation study"""
    print("=" * 60)
    print("PHASE 3: ABLATION STUDY")
    print("Testing which physics components matter most")
    print("=" * 60)
    
    # Load subset of graphs (30 for quick evaluation)
    graphs = load_subset_graphs(num_proteins=30)
    
    if len(graphs) < 20:
        print("Error: Need at least 20 graphs for ablation study")
        return
    
    # Define ablation conditions
    ablation_configs = [
        {
            'name': 'full_physics',
            'disable_components': [],
            'description': 'All physics components enabled (baseline)'
        },
        {
            'name': 'no_electrostatics',
            'disable_components': ['electrostatics'],
            'description': 'Without electrostatic interactions'
        },
        {
            'name': 'no_vdw',
            'disable_components': ['vdw'],
            'description': 'Without van der Waals interactions'
        },
        {
            'name': 'no_hydrogen_bonds',
            'disable_components': ['hydrogen_bonds'],
            'description': 'Without hydrogen bond potential'
        },
        {
            'name': 'no_hydrophobic',
            'disable_components': ['hydrophobic'],
            'description': 'Without hydrophobic interactions'
        },
        {
            'name': 'no_physics',
            'disable_components': ['electrostatics', 'vdw', 'hydrogen_bonds', 'hydrophobic'],
            'description': 'No physics constraints (Base GNN)'
        }
    ]
    
    results = []
    
    # Run each configuration
    for config in tqdm(ablation_configs, desc="Running ablation study"):
        result = train_and_evaluate(config, graphs, config['name'])
        results.append(result)
        
        print(f"  {config['name']}: F1 = {result['best_f1']:.4f}")
    
    # Sort by F1 score
    results.sort(key=lambda x: x['best_f1'], reverse=True)
    
    # Save results
    output_dir = "./experiments/results/phase3/ablation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    with open(f"{output_dir}/ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV for easy reading
    import pandas as pd
    df_data = []
    for r in results:
        row = {
            'model': r['name'],
            'f1_score': r['best_f1'],
            'train_loss': r['final_train_loss'],
            'physics_ratio': r['physics_ratio'],
            'disabled_components': ', '.join(r['disable_components']) if r['disable_components'] else 'none'
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{output_dir}/ablation_results.csv", index=False)
    
    # Create simple summary
    summary = {
        'best_model': results[0]['name'],
        'best_f1': results[0]['best_f1'],
        'worst_model': results[-1]['name'],
        'worst_f1': results[-1]['best_f1'],
        'performance_range': results[0]['best_f1'] - results[-1]['best_f1']
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    for r in results:
        components = r['disable_components']
        comp_str = ', '.join(components) if components else 'all physics'
        print(f"{r['name']:20s} F1={r['best_f1']:.4f} (without: {comp_str})")
    
    print(f"\n✅ Results saved to {output_dir}/")
    
    # Create visualization
    create_ablation_plot(results, output_dir)
    
    return results

def create_ablation_plot(results, output_dir):
    """Create ablation study visualization"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Sort for plotting
    results_sorted = sorted(results, key=lambda x: x['best_f1'])
    names = [r['name'] for r in results_sorted]
    f1_scores = [r['best_f1'] for r in results_sorted]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, f1_scores, color='skyblue')
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center')
    
    plt.xlabel('F1 Score')
    plt.title('Ablation Study: Physics Component Importance')
    plt.xlim(0, max(f1_scores) * 1.15)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_study.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/ablation_study.pdf", bbox_inches='tight')
    
    print(f"\n📊 Plot saved: {output_dir}/ablation_study.png")
    plt.close()

if __name__ == "__main__":
    run_ablation_study()
