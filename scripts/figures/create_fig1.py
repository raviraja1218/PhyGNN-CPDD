#!/usr/bin/env python3
"""
Create Figure 1: Performance Progression
Shows improvement from baseline to PhyGNN
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_fig1():
    """Create Figure 1: Performance progression"""
    print("Creating Figure 1: Performance Progression...")
    
    # Data from our results
    methods = ['Geometric\nBaseline', 'Base GNN\n(No Physics)', 'FPOCKET\n(Literature)', 'PhyGNN\n(Ours)']
    f1_scores = [0.077, 0.308, 0.520, 0.547]
    colors = ['lightgray', 'lightblue', 'orange', 'green']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize plot
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('Performance Progression: Geometric Baseline to PhyGNN', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 0.65)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0.52, color='orange', linestyle='--', alpha=0.5, label='FPOCKET benchmark')
    
    # Add improvement annotations
    plt.annotate('+300%', xy=(0.5, 0.2), xytext=(0.5, 0.25),
                arrowprops=dict(arrowstyle='->', color='blue'),
                ha='center', fontsize=10, fontweight='bold', color='blue')
    
    plt.annotate('+77%', xy=(1.5, 0.4), xytext=(1.5, 0.45),
                arrowprops=dict(arrowstyle='->', color='green'),
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.annotate('+5.2% over\nFPOCKET', xy=(2.5, 0.53), xytext=(2.5, 0.58),
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontsize=9, fontweight='bold', color='red')
    
    # Save figure
    output_dir = "./paper/final_submission/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_performance_progression.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig1_performance_progression.pdf", bbox_inches='tight')
    
    print(f"✓ Figure 1 saved to {output_dir}/")
    plt.close()

if __name__ == "__main__":
    create_fig1()
