import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Check what data we have
phase1_f1 = 0.077
phase2a_f1 = 0.3077
phase2b_f1 = 0.5444
fpocket_f1 = 0.520

# Figure 1: Performance Progression (if missing)
fig1_path = "./paper/final_figures/fig1_performance_progression.png"
if not os.path.exists(fig1_path):
    print("Generating Figure 1...")
    methods = ['Geometric Baseline', 'Base GNN', 'Hamiltonian GNN', 'FPOCKET']
    f1_scores = [phase1_f1, phase2a_f1, phase2b_f1, fpocket_f1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, f1_scores, color=['red', 'orange', 'green', 'blue'])
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14)
    ax.set_ylim(0, 0.6)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("./paper/final_figures/", exist_ok=True)
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {fig1_path}")

# Figure 3: Method Overview (simple version)
fig3_path = "./paper/final_figures/fig3_framework_overview.png"
if not os.path.exists(fig3_path):
    print("Generating Figure 3...")
    # Create simple pipeline diagram
    fig, ax = plt.subplots(figsize=(12, 3))
    
    steps = ['Protein Structure', 'Graph Construction', 'GNN Processing', 
             'Physics Constraints', 'Pocket Prediction']
    
    for i, step in enumerate(steps):
        ax.text(i*2.5, 0, step, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        if i < len(steps)-1:
            ax.arrow(i*2.5+0.5, 0, 1.5, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
    
    ax.set_xlim(-1, len(steps)*2.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('PhyGNN Framework Overview', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {fig3_path}")

print("\nFigure generation complete!")
