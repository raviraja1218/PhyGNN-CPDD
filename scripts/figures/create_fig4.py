#!/usr/bin/env python3
"""
Create Figure 4: Therapeutic Case Studies
STAT3, KRAS, and LRRK2 analyses
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_fig4():
    """Create Figure 4: Therapeutic case studies"""
    print("Creating Figure 4: Therapeutic Case Studies...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Case study data
    proteins = ['STAT3', 'KRAS G12C', 'LRRK2']
    diseases = ['Cancer Immunotherapy', 'Oncology', "Parkinson's Disease"]
    residues = [770, 188, 2527]
    pockets = [46, 15, 88]
    percentages = [6.0, 8.0, 3.5]
    druggability = ['Undruggable → Druggable', 'Mutation-specific', 'Multi-domain']
    
    # Create subplots
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Main bar chart
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(proteins))
    width = 0.25
    
    # Bars for different metrics
    bars1 = ax1.bar(x - width, residues, width, label='Total Residues', color='lightblue')
    bars2 = ax1.bar(x, pockets, width, label='Predicted Pockets', color='lightgreen')
    bars3 = ax1.bar(x + width, percentages, width, label='Pocket %', color='orange')
    
    ax1.set_xlabel('Protein Target', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count / Percentage', fontsize=12, fontweight='bold')
    ax1.set_title('Case Study Analysis: Therapeutic Targets', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{p}\n({d})' for p, d in zip(proteins, diseases)], fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'{int(height) if height > 1 else f"{height:.1f}"}',
                    ha='center', va='bottom', fontsize=9)
    
    # Protein schematics (simplified)
    ax2 = fig.add_subplot(gs[1, 0])
    # STAT3 schematic
    ax2.text(0.5, 0.8, 'STAT3', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.6, 'Transcription Factor', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.4, 'Flat PPIs → Pocket Discovery', ha='center', va='center', fontsize=9, style='italic')
    ax2.text(0.5, 0.2, 'Cancer Immunotherapy', ha='center', va='center', fontsize=9, color='red')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 1])
    # KRAS schematic
    ax3.text(0.5, 0.8, 'KRAS G12C', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.6, 'Oncogene', ha='center', va='center', fontsize=10)
    ax3.text(0.5, 0.4, 'Mutation-induced Pocket', ha='center', va='center', fontsize=9, style='italic')
    ax3.text(0.5, 0.2, 'Historically "Undruggable"', ha='center', va='center', fontsize=9, color='red')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 2])
    # LRRK2 schematic
    ax4.text(0.5, 0.8, 'LRRK2', ha='center', va='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.6, 'Kinase', ha='center', va='center', fontsize=10)
    ax4.text(0.5, 0.4, 'Large Protein (2527 residues)', ha='center', va='center', fontsize=9, style='italic')
    ax4.text(0.5, 0.2, "Parkinson's Disease", ha='center', va='center', fontsize=9, color='red')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Add summary text
    summary_text = """Key Insights:
• STAT3: From "undruggable" to potentially druggable via cryptic pocket discovery
• KRAS G12C: Method validates known drug binding sites and identifies new possibilities
• LRRK2: Demonstrates scalability to large proteins with complex domain structures
• All cases: Physics-informed predictions align with therapeutic relevance"""
    
    plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
    
    # Save figure
    output_dir = "./paper/final_submission/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(f"{output_dir}/fig4_case_studies.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig4_case_studies.pdf", bbox_inches='tight')
    
    print(f"✓ Figure 4 saved to {output_dir}/")
    plt.close()

if __name__ == "__main__":
    create_fig4()
