#!/usr/bin/env python3
"""
Analyze physics features in existing graphs
"""
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

def analyze_features():
    """Analyze what features exist in our graphs"""
    print("="*60)
    print("PHASE 3: PHYSICS FEATURE ANALYSIS")
    print("="*60)
    
    # Use physics graphs from Phase 2B
    data_dir = "./data/processed/physics_graphs/train"
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')][:10]  # First 10
    print(f"Analyzing {len(graph_files)} physics graphs from {data_dir}")
    
    all_features = []
    feature_stats = {}
    
    for i, gf in enumerate(graph_files):
        try:
            graph = torch.load(os.path.join(data_dir, gf), weights_only=True)
            
            if hasattr(graph, 'x'):
                n_features = graph.x.shape[1]
                all_features.append(n_features)
                
                # Record stats for first graph
                if i == 0:
                    print(f"\n📊 Sample graph: {gf}")
                    print(f"   Nodes: {graph.num_nodes}")
                    print(f"   Edges: {graph.edge_index.shape[1]}")
                    print(f"   Features: {n_features} dimensions")
                    print(f"   Labels: {graph.y.sum().item()}/{graph.num_nodes} positive")
                    
                    # Analyze feature columns
                    for col in range(min(10, n_features)):
                        data = graph.x[:, col].numpy()
                        print(f"   Feature {col:2d}: mean={np.mean(data):.3f}, "
                              f"std={np.std(data):.3f}")
        
        except Exception as e:
            print(f"  Error loading {gf}: {e}")
    
    # Summary
    if all_features:
        print(f"\n📈 Feature Dimension Summary:")
        print(f"   Average features: {np.mean(all_features):.1f}")
        print(f"   Range: {min(all_features)} to {max(all_features)}")
    
    # Check for specific physics attributes
    print(f"\n🔍 Checking for Physics Attributes:")
    
    # Load one graph and check attributes
    if graph_files:
        sample_graph = torch.load(os.path.join(data_dir, graph_files[0]), weights_only=True)
        
        # List all attributes
        attrs = dir(sample_graph)
        physics_keywords = ['charge', 'vdw', 'hbond', 'hydrophob', 'electro', 'physics', 'energy', 'bond', 'angle']
        
        physics_attrs = []
        for attr in attrs:
            attr_lower = attr.lower()
            if any(keyword in attr_lower for keyword in physics_keywords):
                if not attr.startswith('_'):
                    physics_attrs.append(attr)
        
        if physics_attrs:
            print(f"   Found {len(physics_attrs)} physics-related attributes:")
            for attr in sorted(physics_attrs)[:10]:  # Show first 10
                val = getattr(sample_graph, attr)
                if isinstance(val, torch.Tensor):
                    print(f"     {attr}: shape={val.shape}")
                else:
                    print(f"     {attr}: {type(val)}")
        else:
            print("   No explicit physics attributes found")
            
        # Check feature names if available
        if hasattr(sample_graph, 'feature_names'):
            print(f"\n📝 Feature Names:")
            for i, name in enumerate(sample_graph.feature_names[:15]):  # First 15
                print(f"   {i:2d}: {name}")
    
    # Save analysis
    output_dir = "./experiments/results/phase3/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    analysis = {
        'n_graphs_analyzed': len(graph_files),
        'feature_dimensions': all_features,
        'avg_features': float(np.mean(all_features)) if all_features else 0,
        'sample_graph_stats': {
            'nodes': int(sample_graph.num_nodes) if 'sample_graph' in locals() else 0,
            'edges': int(sample_graph.edge_index.shape[1]) if 'sample_graph' in locals() else 0,
            'features': int(sample_graph.x.shape[1]) if 'sample_graph' in locals() and hasattr(sample_graph, 'x') else 0
        }
    }
    
    with open(f"{output_dir}/feature_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis saved to {output_dir}/feature_analysis.json")
    
    # Create visualization
    create_feature_visualization(sample_graph, output_dir)
    
    return analysis

def create_feature_visualization(graph, output_dir):
    """Create feature visualization"""
    if not hasattr(graph, 'x'):
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot first 20 features
    n_features_to_plot = min(20, graph.x.shape[1])
    
    for i in range(n_features_to_plot):
        plt.subplot(4, 5, i+1)
        data = graph.x[:, i].numpy()
        plt.hist(data, bins=30, alpha=0.7)
        plt.title(f'Feature {i}')
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle(f'Feature Distributions (First {n_features_to_plot} Features)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
    
    print(f"📊 Feature distribution plot saved")
    plt.close()

if __name__ == "__main__":
    analyze_features()
