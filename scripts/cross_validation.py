#!/usr/bin/env python3
"""
5-fold Cross-Validation for PhyGNN
Statistical validation of our method
"""
import os
import sys
import torch
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import time
from tqdm import tqdm

# Add our modules to path
sys.path.append('./src/models')
from hamiltonian_gnn_phase2c import HamiltonianGNN
from physics_trainer import PhysicsTrainer
from torch_geometric.loader import DataLoader

def load_all_graphs():
    """Load all processed physics graphs"""
    print("Loading all processed graphs...")
    
    graphs = []
    graph_dir = "./data/processed/physics_graphs/train/"
    
    if not os.path.exists(graph_dir):
        print(f"Error: Graph directory not found at {graph_dir}")
        return []
    
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    print(f"Found {len(graph_files)} graph files")
    
    for gf in tqdm(graph_files, desc="Loading graphs"):
        try:
            graph = torch.load(os.path.join(graph_dir, gf), weights_only=True)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Could not load {gf}: {e}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs

def run_kfold_cross_validation(graphs, n_folds=5, epochs=50):
    """
    Run k-fold cross-validation
    
    Args:
        graphs: List of PyG graphs
        n_folds: Number of folds (default: 5)
        epochs: Training epochs per fold
    
    Returns:
        Dictionary with CV results
    """
    print(f"\nRunning {n_folds}-fold cross-validation...")
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_fold_metrics = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'auc_scores': [],
        'training_times': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(graphs), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")
        
        fold_start_time = time.time()
        
        # Split data
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        
        print(f"  Training samples: {len(train_graphs)}")
        print(f"  Validation samples: {len(val_graphs)}")
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
        
        # Create model
        input_dim = train_graphs[0].x.shape[1]
        model = HamiltonianGNN(
            input_dim=input_dim,
            hidden_dim=128,
            lambda_physics=0.0001
        )
        
        # Create trainer
        trainer = PhysicsTrainer(
            model=model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=0.001,
            pos_weight=20.0
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_dir=None,  # Don't save models during CV
            verbose=False
        )
        
        # Evaluate
        val_loss, val_f1, val_precision, val_recall, val_auc = trainer.evaluate(val_loader)
        
        fold_time = time.time() - fold_start_time
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'train_samples': len(train_graphs),
            'val_samples': len(val_graphs),
            'best_val_f1': max(history['val_f1']) if history['val_f1'] else 0,
            'final_val_f1': val_f1,
            'final_val_precision': val_precision,
            'final_val_recall': val_recall,
            'final_val_auc': val_auc,
            'training_time_seconds': fold_time
        }
        
        fold_results.append(fold_result)
        
        # Update overall metrics
        all_fold_metrics['f1_scores'].append(val_f1)
        all_fold_metrics['precision_scores'].append(val_precision)
        all_fold_metrics['recall_scores'].append(val_recall)
        all_fold_metrics['auc_scores'].append(val_auc)
        all_fold_metrics['training_times'].append(fold_time)
        
        print(f"  Results:")
        print(f"    F1: {val_f1:.4f}")
        print(f"    Precision: {val_precision:.4f}")
        print(f"    Recall: {val_recall:.4f}")
        print(f"    AUC: {val_auc:.4f}")
        print(f"    Training time: {fold_time:.1f}s")
    
    return fold_results, all_fold_metrics

def calculate_statistics(metrics):
    """Calculate statistical summaries"""
    stats = {}
    
    for metric_name, values in metrics.items():
        if values:
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                '95_ci_lower': np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values)),
                '95_ci_upper': np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values))
            }
    
    return stats

def save_cv_results(fold_results, overall_metrics, stats):
    """Save cross-validation results"""
    output_dir = "./experiments/results/phase4/cross_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fold-by-fold results
    with open(os.path.join(output_dir, 'cv_results_5fold.json'), 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'statistics': stats,
            'cv_metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'n_folds': 5,
                'total_samples': len(overall_metrics['f1_scores']) * 2  # rough estimate
            }
        }, f, indent=2)
    
    # Create LaTeX table
    latex_table = create_latex_table(fold_results, stats)
    with open(os.path.join(output_dir, 'statistical_tests.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f"\n✓ CV results saved to {output_dir}")

def create_latex_table(fold_results, stats):
    """Create LaTeX table for paper"""
    latex = """\\begin{table}[htbp]
\\centering
\\caption{5-Fold Cross-Validation Results for PhyGNN}
\\label{tab:cv_results}
\\begin{tabular}{cccccc}
\\hline
\\textbf{Fold} & \\textbf{F1 Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{AUC} & \\textbf{Time (s)} \\\\
\\hline
"""
    
    for fold in fold_results:
        latex += f"{fold['fold']} & {fold['final_val_f1']:.3f} & {fold['final_val_precision']:.3f} & {fold['final_val_recall']:.3f} & {fold['final_val_auc']:.3f} & {fold['training_time_seconds']:.0f} \\\\\n"
    
    latex += f"\\hline\n\\textbf{{Mean}} & {stats['f1_scores']['mean']:.3f} & {stats['precision_scores']['mean']:.3f} & {stats['recall_scores']['mean']:.3f} & {stats['auc_scores']['mean']:.3f} & {stats['training_times']['mean']:.0f} \\\\\n"
    latex += f"\\textbf{{Std}} & {stats['f1_scores']['std']:.3f} & {stats['precision_scores']['std']:.3f} & {stats['recall_scores']['std']:.3f} & {stats['auc_scores']['std']:.3f} & {stats['training_times']['std']:.0f} \\\\\n"
    latex += f"\\textbf{{95\\% CI}} & [{stats['f1_scores']['95_ci_lower']:.3f}, {stats['f1_scores']['95_ci_upper']:.3f}] & [{stats['precision_scores']['95_ci_lower']:.3f}, {stats['precision_scores']['95_ci_upper']:.3f}] & [{stats['recall_scores']['95_ci_lower']:.3f}, {stats['recall_scores']['95_ci_upper']:.3f}] & [{stats['auc_scores']['95_ci_lower']:.3f}, {stats['auc_scores']['95_ci_upper']:.3f}] & - \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}"""
    
    return latex

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("PHYGNN 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Load graphs
    graphs = load_all_graphs()
    
    if len(graphs) < 10:
        print("Error: Need at least 10 graphs for cross-validation")
        return
    
    # Run cross-validation
    fold_results, overall_metrics = run_kfold_cross_validation(graphs, n_folds=5, epochs=30)
    
    # Calculate statistics
    stats = calculate_statistics(overall_metrics)
    
    # Save results
    save_cv_results(fold_results, overall_metrics, stats)
    
    # Print summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Mean F1: {stats['f1_scores']['mean']:.4f} ± {stats['f1_scores']['std']:.4f}")
    print(f"95% CI for F1: [{stats['f1_scores']['95_ci_lower']:.4f}, {stats['f1_scores']['95_ci_upper']:.4f}]")
    print(f"Mean Precision: {stats['precision_scores']['mean']:.4f} ± {stats['precision_scores']['std']:.4f}")
    print(f"Mean Recall: {stats['recall_scores']['mean']:.4f} ± {stats['recall_scores']['std']:.4f}")
    print(f"Mean AUC: {stats['auc_scores']['mean']:.4f} ± {stats['auc_scores']['std']:.4f}")
    print(f"Mean training time per fold: {stats['training_times']['mean']:.1f}s")
    
    # Compare with baseline
    baseline_f1 = 0.077
    improvement = (stats['f1_scores']['mean'] - baseline_f1) / baseline_f1 * 100
    print(f"\nImprovement over baseline ({baseline_f1:.3f}): {improvement:.1f}%")
    
    print(f"\n✓ Cross-validation complete!")

if __name__ == "__main__":
    main()
