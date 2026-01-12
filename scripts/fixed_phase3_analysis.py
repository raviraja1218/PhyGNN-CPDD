#!/usr/bin/env python3
"""
Fixed Phase 3 analysis with proper torch.load settings
"""
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Add Data to safe globals for weights_only loading
torch.serialization.add_safe_globals([Data])

def safe_load_graph(filepath):
    """Safely load graph with proper settings"""
    try:
        # Try weights_only=True first (safer)
        graph = torch.load(filepath, weights_only=True)
        return graph
    except:
        try:
            # If that fails, try without weights_only (less safe but works)
            print(f"  Using weights_only=False for {os.path.basename(filepath)}")
            graph = torch.load(filepath, weights_only=False)
            return graph
        except Exception as e:
            print(f"  Failed to load {filepath}: {e}")
            return None

def analyze_what_we_have():
    """Analyze what data we actually have"""
    print("="*60)
    print("PHASE 3: DATA INVENTORY ANALYSIS")
    print("="*60)
    
    # Check all data directories
    data_dirs = [
        "./data/processed/physics_graphs/train/",
        "./data/processed/graphs_working/train/",
        "./data/processed/phase2c_final_300_converted/train/",
        "./data/processed/graphs/train/"
    ]
    
    inventory = {}
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
            if pt_files:
                # Try to load first file to check structure
                sample_file = os.path.join(dir_path, pt_files[0])
                sample_graph = safe_load_graph(sample_file)
                
                if sample_graph is not None:
                    inventory[dir_path] = {
                        'n_files': len(pt_files),
                        'sample_stats': {
                            'nodes': int(sample_graph.num_nodes) if hasattr(sample_graph, 'num_nodes') else 0,
                            'edges': int(sample_graph.edge_index.shape[1]) if hasattr(sample_graph, 'edge_index') else 0,
                            'features': int(sample_graph.x.shape[1]) if hasattr(sample_graph, 'x') else 0,
                            'has_labels': hasattr(sample_graph, 'y')
                        },
                        'files': pt_files[:3]  # First 3 filenames
                    }
                    
                    print(f"\n📁 {dir_path}:")
                    print(f"   Files: {len(pt_files)}")
                    print(f"   Sample: {pt_files[0]}")
                    print(f"     Nodes: {sample_graph.num_nodes if hasattr(sample_graph, 'num_nodes') else 'N/A'}")
                    print(f"     Edges: {sample_graph.edge_index.shape[1] if hasattr(sample_graph, 'edge_index') else 'N/A'}")
                    print(f"     Features: {sample_graph.x.shape[1] if hasattr(sample_graph, 'x') else 'N/A'}")
                    print(f"     Labels: {'Yes' if hasattr(sample_graph, 'y') else 'No'}")
                else:
                    inventory[dir_path] = {
                        'n_files': len(pt_files),
                        'error': 'Failed to load sample'
                    }
    
    # Save inventory
    output_dir = "./experiments/results/phase3"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/data_inventory.json", 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n✅ Inventory saved to {output_dir}/data_inventory.json")
    
    # Determine which dataset to use
    print("\n" + "="*60)
    print("RECOMMENDED DATASET FOR ANALYSIS:")
    print("="*60)
    
    best_dir = None
    for dir_path, info in inventory.items():
        if 'sample_stats' in info:
            stats = info['sample_stats']
            if stats['features'] > 20 and stats['has_labels']:
                print(f"✓ {dir_path}: {info['n_files']} files, {stats['features']} features")
                if best_dir is None or info['n_files'] > inventory.get(best_dir, {}).get('n_files', 0):
                    best_dir = dir_path
            else:
                print(f"✗ {dir_path}: Missing features or labels")
    
    if best_dir:
        print(f"\n🎯 USE: {best_dir}")
        print(f"   Reason: Has {inventory[best_dir]['n_files']} files with features and labels")
        return best_dir, inventory[best_dir]
    else:
        print("\n❌ No suitable dataset found!")
        return None, None

def quick_ablation_analysis(data_dir, inventory_info):
    """Quick analysis without retraining"""
    print("\n" + "="*60)
    print("QUICK ABLATION ANALYSIS")
    print("="*60)
    
    if not data_dir:
        print("No data directory provided")
        return
    
    # Load a few graphs
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')][:10]
    
    feature_stats = []
    
    for pf in pt_files:
        graph = safe_load_graph(os.path.join(data_dir, pf))
        if graph is not None and hasattr(graph, 'x'):
            features = graph.x.numpy()
            stats = {
                'protein': pf.replace('.pt', ''),
                'n_features': features.shape[1],
                'feature_means': features.mean(axis=0).tolist(),
                'feature_stds': features.std(axis=0).tolist()
            }
            feature_stats.append(stats)
    
    if feature_stats:
        # Analyze which features vary the most
        n_features = feature_stats[0]['n_features']
        feature_variability = []
        
        for i in range(min(20, n_features)):  # First 20 features
            means = [s['feature_means'][i] for s in feature_stats]
            variability = np.std(means)  # How much this feature varies across proteins
            
            feature_variability.append({
                'index': i,
                'variability': float(variability),
                'avg_mean': float(np.mean(means))
            })
        
        # Sort by variability
        feature_variability.sort(key=lambda x: x['variability'], reverse=True)
        
        print("\n📊 Feature Variability (Higher = More Important):")
        print("Rank | Feature | Variability | Avg Value")
        print("-" * 50)
        
        for i, fv in enumerate(feature_variability[:10]):  # Top 10
            # Try to guess feature type
            if fv['index'] < 20:
                feat_name = f"Residue_Type_{fv['index']}"
            elif fv['index'] == 20:
                feat_name = "Partial_Charge"
            elif fv['index'] == 21:
                feat_name = "VDW_Radius"
            elif fv['index'] == 22:
                feat_name = "HB_Donor"
            elif fv['index'] == 23:
                feat_name = "HB_Acceptor"
            elif fv['index'] == 24:
                feat_name = "Hydrophobicity"
            else:
                feat_name = f"Feature_{fv['index']}"
            
            print(f"{i+1:4d} | {feat_name:15s} | {fv['variability']:10.4f} | {fv['avg_mean']:8.3f}")
        
        # Save results
        output_dir = "./experiments/results/phase3/ablation"
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'analysis_type': 'feature_variability_ablation',
            'data_dir': data_dir,
            'n_proteins_analyzed': len(feature_stats),
            'n_features': n_features,
            'top_features': feature_variability[:15]
        }
        
        with open(f"{output_dir}/feature_variability.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Feature variability analysis saved")
        
        # Create plot
        create_variability_plot(feature_variability[:15], output_dir)
    
    return feature_stats

def create_variability_plot(feature_variability, output_dir):
    """Create feature variability plot"""
    indices = [fv['index'] for fv in feature_variability]
    variabilities = [fv['variability'] for fv in feature_variability]
    
    # Create feature names
    feature_names = []
    for idx in indices:
        if idx < 20:
            feature_names.append(f"Residue_{idx}")
        elif idx == 20:
            feature_names.append("Charge")
        elif idx == 21:
            feature_names.append("VDW")
        elif idx == 22:
            feature_names.append("HB_Donor")
        elif idx == 23:
            feature_names.append("HB_Acceptor")
        elif idx == 24:
            feature_names.append("Hydrophob")
        else:
            feature_names.append(f"F_{idx}")
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(feature_names)), variabilities, color='skyblue', alpha=0.7)
    
    # Add value labels
    for bar, var in zip(bars, variabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{var:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.xlabel('Feature Index')
    plt.ylabel('Variability (Std of Means)')
    plt.title('Feature Variability Across Proteins')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/feature_variability.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Variability plot saved: {output_dir}/feature_variability.png")

if __name__ == "__main__":
    # Step 1: Analyze what we have
    best_dir, inventory_info = analyze_what_we_have()
    
    # Step 2: Quick analysis
    if best_dir:
        quick_ablation_analysis(best_dir, inventory_info)
    
    print("\n" + "="*60)
    print("PHASE 3 SIMPLIFIED ANALYSIS COMPLETE")
    print("="*60)
