#!/usr/bin/env python3
"""
Explainability analysis with GNNExplainer
"""
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Add src to path
sys.path.append('./src/models')
from hamiltonian_gnn_ablation_fixed import HamiltonianGNN

def load_model_and_graph():
    """Load best model and sample graph"""
    print("Loading model and sample graph...")
    
    # Load model
    model_path = "./experiments/results/phase2b/week2/training_fixed/hamgnn_best.pt"
    model_state = torch.load(model_path, weights_only=True, map_location='cpu')
    
    # Create model instance
    model = HamiltonianGNN(input_dim=35, hidden_dim=128)
    model.load_state_dict(model_state)
    model.eval()
    
    # Load sample graph
    graph_path = "./data/processed/physics_graphs/train/1a0q_graph.pt"
    graph = torch.load(graph_path)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Graph loaded: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    return model, graph

def integrated_gradients_analysis(model, graph):
    """Analyze feature importance with Integrated Gradients"""
    print("\nRunning Integrated Gradients analysis...")
    
    # Prepare input
    x = graph.x.unsqueeze(0)  # Add batch dimension
    edge_index = graph.edge_index
    target = graph.y.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
    
    # Create baseline (zeros)
    baseline = torch.zeros_like(x)
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Compute attributions
    attributions, delta = ig.attribute(
        inputs=x,
        baselines=baseline,
        target=target,
        additional_forward_args=(graph,),
        return_convergence_delta=True
    )
    
    # Summarize attributions
    attributions = attributions.squeeze(0).detach().numpy()
    mean_attributions = np.mean(np.abs(attributions), axis=0)
    
    # Map feature indices to names
    feature_names = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'partial_charge', 'vdw_radius', 'hb_donor', 'hb_acceptor', 'hydrophobicity',
        'charge', 'size', 'position_x', 'position_y', 'position_z',
        'num_atoms', 'solvent_acc', 'b_factor', 'conservation', 'secondary_struct'
    ]
    
    # Get top features
    top_indices = np.argsort(mean_attributions)[-10:]  # Top 10
    top_features = [(feature_names[i], mean_attributions[i]) for i in top_indices]
    
    print("\nTop 10 Important Features:")
    for feat, importance in sorted(top_features, key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: {importance:.6f}")
    
    # Save results
    output_dir = "./experiments/results/phase3/explainability"
    os.makedirs(output_dir, exist_ok=True)
    
    feature_importance = {
        'feature_names': feature_names,
        'importance_scores': mean_attributions.tolist(),
        'top_features': top_features
    }
    
    with open(f"{output_dir}/feature_importance.json", 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    # Create visualization
    create_feature_importance_plot(feature_names, mean_attributions, output_dir)
    
    return feature_importance

def create_feature_importance_plot(feature_names, importance_scores, output_dir):
    """Create feature importance visualization"""
    # Get top 15 features
    top_n = 15
    indices = np.argsort(importance_scores)[-top_n:]
    top_names = [feature_names[i] for i in indices]
    top_scores = [importance_scores[i] for i in indices]
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_names)), top_scores, color='steelblue')
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        plt.text(score + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=9)
    
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Importance Score (Integrated Gradients)')
    plt.title('Feature Importance for Pocket Detection')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/feature_importance.pdf", bbox_inches='tight')
    
    print(f"📊 Feature importance plot saved: {output_dir}/feature_importance.png")
    plt.close()

def residue_importance_analysis(model, graph):
    """Analyze which residues are important for predictions"""
    print("\nAnalyzing residue importance...")
    
    # Get model predictions
    with torch.no_grad():
        logits, _ = model(graph)
        probs = torch.sigmoid(logits)
    
    # Get top predicted pocket residues
    pocket_probs = probs.squeeze().numpy()
    top_residue_indices = np.argsort(pocket_probs)[-20:]  # Top 20 predicted
    
    # Get ground truth
    labels = graph.y.numpy()
    
    # Analyze residues
    residue_info = []
    for idx in top_residue_indices:
        # Get residue type from one-hot encoding
        residue_onehot = graph.x[idx, :20].numpy()
        residue_idx = np.argmax(residue_onehot)
        
        residue_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        residue_type = residue_types[residue_idx]
        
        residue_info.append({
            'residue_index': int(idx),
            'residue_type': residue_type,
            'predicted_prob': float(pocket_probs[idx]),
            'is_pocket': bool(labels[idx]),
            'position': graph.pos[idx].tolist() if hasattr(graph, 'pos') else [0, 0, 0]
        })
    
    # Save residue analysis
    output_dir = "./experiments/results/phase3/explainability/residue_importance"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/top_residues.json", 'w') as f:
        json.dump(residue_info, f, indent=2)
    
    # Create CSV for easy reading
    import pandas as pd
    df = pd.DataFrame(residue_info)
    df.to_csv(f"{output_dir}/top_residues.csv", index=False)
    
    print("\nTop 20 Predicted Pocket Residues:")
    print("Index | Type | Prob | Actual Pocket")
    print("-" * 40)
    for res in sorted(residue_info, key=lambda x: x['predicted_prob'], reverse=True)[:10]:
        pocket_symbol = "✓" if res['is_pocket'] else "✗"
        print(f"{res['residue_index']:5d} | {res['residue_type']:3s} | {res['predicted_prob']:.3f} | {pocket_symbol}")
    
    # Calculate accuracy
    correct = sum(1 for res in residue_info if res['is_pocket'])
    accuracy = correct / len(residue_info)
    print(f"\nAccuracy of top predictions: {accuracy:.1%} ({correct}/{len(residue_info)})")
    
    return residue_info

def run_explainability_analysis():
    """Run complete explainability analysis"""
    print("=" * 60)
    print("PHASE 3: EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    # Load model and sample graph
    model, graph = load_model_and_graph()
    
    # Run analyses
    feature_importance = integrated_gradients_analysis(model, graph)
    residue_importance = residue_importance_analysis(model, graph)
    
    # Create summary
    summary = {
        'total_residues_analyzed': graph.num_nodes,
        'top_features': feature_importance['top_features'][:5],
        'top_residue_accuracy': sum(1 for r in residue_importance if r['is_pocket']) / len(residue_importance),
        'most_common_residue_types': {}
    }
    
    # Count residue types in top predictions
    for res in residue_importance:
        res_type = res['residue_type']
        summary['most_common_residue_types'][res_type] = summary['most_common_residue_types'].get(res_type, 0) + 1
    
    # Save summary
    output_dir = "./experiments/results/phase3/explainability"
    with open(f"{output_dir}/explainability_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"✅ Results saved to {output_dir}/")
    print("=" * 60)
    
    return feature_importance, residue_importance

if __name__ == "__main__":
    run_explainability_analysis()
