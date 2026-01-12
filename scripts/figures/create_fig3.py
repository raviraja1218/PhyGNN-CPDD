#!/usr/bin/env python3
"""
Create Figure 3: Physics Impact Analysis
Ablation study showing component contributions
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_fig3():
    """Create Figure 3: Physics impact analysis"""
    print("Creating Figure 3: Physics Impact Analysis...")
    
    # Ablation study data (from Phase 3)
    components = ['Full Physics', 'No Electrostatics', 'No van der Waals', 
                  'No H-Bonds', 'No Hydrophobic', 'No Physics']
    f1_scores = [0.544, 0.488, 0.512, 0.531, 0.525, 0.308]
    drops = [0, -10.4, -6.0, -2.5, -3.6, -43.5]  # percentage drops
    
    # Colors based on performance
    colors = ['green', 'orange', 'orange', 'lightgreen', 'lightgreen', 'red']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Bar chart of F1 scores
    bars1 = ax1.bar(components, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study: Impact of Physics Components', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 0.65)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add drop percentages
    for i, (bar, drop) in enumerate(zip(bars1, drops)):
        if drop < 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{drop:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
    
    # Subplot 2: Physics contribution pie chart
    physics_contributions = {
        'Electrostatics': 10.4,
        'van der Waals': 6.0,
        'Hydrogen Bonds': 2.5,
        'Hydrophobic': 3.6,
        'Other Physics': 4.4,
        'Base GNN': 30.8
    }
    
    labels = list(physics_contributions.keys())
    sizes = list(physics_contributions.values())
    colors_pie = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue', 'gray']
    explode = (0.1, 0.05, 0, 0, 0, 0)  # explode electrostatics
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Relative Contribution to Performance Improvement', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    
    # Add annotation
    plt.figtext(0.5, 0.02, 'Key Finding: Electrostatics is the most important physics component (10.4% of F1 improvement)',
                ha='center', fontsize=11, fontweight='bold', style='italic')
    
    # Save figure
    output_dir = "./paper/final_submission/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(f"{output_dir}/fig3_physics_impact.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig3_physics_impact.pdf", bbox_inches='tight')
    
    print(f"✓ Figure 3 saved to {output_dir}/")
    plt.close()

if __name__ == "__main__":
    create_fig3()
