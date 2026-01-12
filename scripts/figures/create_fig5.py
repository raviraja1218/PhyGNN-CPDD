#!/usr/bin/env python3
"""
Create Figure 5: Benchmarking Comparison
Radar chart and performance comparison
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_fig5():
    """Create Figure 5: Benchmarking comparison"""
    print("Creating Figure 5: Benchmarking Comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Radar chart comparison
    methods = ['PhyGNN (Ours)', 'FPOCKET', 'Base GNN', 'Geometric']
    metrics = ['F1 Score', 'Precision', 'Recall', 'AUC', 'Physics', 'Speed']
    
    # Normalized scores (0-1)
    data = np.array([
        [0.547/0.547, 0.630/0.630, 0.479/0.866, 0.926/0.926, 1.0, 65/120],  # PhyGNN
        [0.520/0.547, 0.480/0.630, 0.570/0.866, 0.670/0.926, 0.0, 120/120], # FPOCKET
        [0.308/0.547, 0.210/0.630, 0.560/0.866, 0.750/0.926, 0.0, 45/120],  # Base GNN
        [0.077/0.547, 0.030/0.630, 0.866/0.866, 0.500/0.926, 0.0, 5/120]    # Geometric
    ])
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    data = np.concatenate((data, data[:,[0]]), axis=1)
    angles += angles[:1]
    
    colors = ['green', 'orange', 'blue', 'gray']
    labels = methods
    
    for idx, (color, label) in enumerate(zip(colors, labels)):
        ax1.plot(angles, data[idx], 'o-', linewidth=2, color=color, label=label)
        ax1.fill(angles, data[idx], alpha=0.1, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Multi-metric Comparison (Normalized)', fontsize=13, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # Subplot 2: Bar chart with improvements
    comparisons = ['vs Geometric', 'vs Base GNN', 'vs FPOCKET']
    improvements = [610.6, 77.7, 5.2]  # percentage improvements
    colors_bar = ['lightblue', 'lightgreen', 'orange']
    
    bars = ax2.bar(comparisons, improvements, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Improvements of PhyGNN', fontsize=13, fontweight='bold', pad=20)
    ax2.set_ylim(0, 650)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 10,
                f'{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add annotations
    ax2.text(0.5, 400, '607% total improvement\nfrom baseline', 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    ax2.text(1.5, 150, 'Physics provides\n77.7% improvement', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
    
    ax2.text(2.5, 80, 'State-of-the-art:\n5.2% over FPOCKET', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
    
    # Overall title
    plt.suptitle('Benchmarking: PhyGNN vs State-of-the-Art Methods', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save figure
    output_dir = "./paper/final_submission/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_benchmarking.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig5_benchmarking.pdf", bbox_inches='tight')
    
    print(f"✓ Figure 5 saved to {output_dir}/")
    plt.close()

if __name__ == "__main__":
    create_fig5()
