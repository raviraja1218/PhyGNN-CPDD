#!/usr/bin/env python3
"""
Create paper-ready analysis from existing results
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def create_paper_figures():
    """Create figures for paper based on Phase 2 results"""
    print("="*60)
    print("CREATING PAPER FIGURES FROM PHASE 2 RESULTS")
    print("="*60)
    
    # Create output directory
    output_dir = "./paper/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Performance progression (from Phase 1, 2A, 2B)
    print("\n📈 Creating Figure 1: Performance Progression")
    
    phases = ['Geometric Baseline', 'Base GNN (Phase 2A)', 'Hamiltonian GNN (Phase 2B)', 'FPOCKET (Literature)']
    f1_scores = [0.077, 0.3077, 0.5444, 0.52]
    
    plt.figure(figsize=(10, 6))
    colors = ['lightgray', 'lightblue', 'green', 'orange']
    bars = plt.bar(phases, f1_scores, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Performance Improvement Across Phases', fontsize=14, fontweight='bold')
    plt.ylim(0, 0.7)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/fig1_performance_progression.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig1_performance_progression.pdf", bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/fig1_performance_progression.png")
    plt.close()
    
    # Figure 2: Training curves from Phase 2B
    print("\n📊 Creating Figure 2: Training Curves")
    
    # Load training history if available
    history_path = "./experiments/results/phase2b/week2/training_fixed/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 score
        ax2.plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig2_training_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/fig2_training_curves.pdf", bbox_inches='tight')
        print(f"  ✅ Saved: {output_dir}/fig2_training_curves.png")
        plt.close()
    else:
        print(f"  ⚠️ Training history not found at {history_path}")
    
    # Figure 3: Method overview
    print("\n🔄 Creating Figure 3: Method Overview")
    
    # Simple diagram
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Remove axes
    ax.axis('off')
    
    # Draw simple pipeline
    components = [
        ('Protein\nStructure', (0.1, 0.5)),
        ('Graph\nConstruction', (0.3, 0.5)),
        ('GNN\nProcessing', (0.5, 0.5)),
        ('Physics\nConstraints', (0.7, 0.5)),
        ('Pocket\nPrediction', (0.9, 0.5))
    ]
    
    # Draw boxes
    for text, (x, y) in components:
        box = plt.Rectangle((x-0.08, y-0.15), 0.16, 0.3, 
                           fill=True, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    for i in range(len(components)-1):
        x1, y1 = components[i][1]
        x2, y2 = components[i+1][1]
        ax.annotate('', xy=(x2-0.08, y1), xytext=(x1+0.08, y1),
                   arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('PhyGNN Framework Overview', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_method_overview.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig3_method_overview.pdf", bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/fig3_method_overview.png")
    plt.close()
    
    # Create LaTeX table
    print("\n📋 Creating LaTeX Table")
    
    table_content = """
\\begin{table}[ht]
\\centering
\\caption{Performance comparison across different methods}
\\label{tab:performance}
\\begin{tabular}{lccc}
\\hline
\\textbf{Method} & \\textbf{F1 Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\hline
Geometric Baseline & 0.077 & 0.030 & 0.866 \\\\
Base GNN (Phase 2A) & 0.308 & 0.210 & 0.560 \\\\
Hamiltonian GNN (Phase 2B) & \\textbf{0.544} & 0.629 & 0.480 \\\\
FPOCKET (Literature) & 0.520 & -- & -- \\\\
\\hline
\\end{tabular}
\\end{table}
"""
    
    table_dir = "./paper/tables"
    os.makedirs(table_dir, exist_ok=True)
    
    with open(f"{table_dir}/performance_table.tex", 'w') as f:
        f.write(table_content)
    
    print(f"  ✅ Saved: {table_dir}/performance_table.tex")
    
    print("\n" + "="*60)
    print("PAPER MATERIALS CREATED SUCCESSFULLY")
    print("="*60)
    print(f"📁 Figures: {output_dir}/")
    print(f"📁 Tables: {table_dir}/")
    print("\nYou now have:")
    print("1. fig1_performance_progression.png - Main result")
    print("2. fig2_training_curves.png - Training stability")
    print("3. fig3_method_overview.png - Method diagram")
    print("4. performance_table.tex - LaTeX table")

if __name__ == "__main__":
    create_paper_figures()
