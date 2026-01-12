#!/usr/bin/env python3
"""
Simple feature importance analysis without model retraining
"""
import os
import torch
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def analyze_feature_importance():
    """Analyze feature importance with simple ML"""
    print("="*60)
    print("PHASE 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load multiple graphs and combine
    data_dir = "./data/processed/physics_graphs/train"
    graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')][:20]  # First 20
    
    if not graph_files:
        print("❌ No graph files found")
        return
    
    all_features = []
    all_labels = []
    
    print(f"Loading {len(graph_files)} graphs...")
    
    for gf in graph_files:
        try:
            graph = torch.load(os.path.join(data_dir, gf), weights_only=True)
            
            if hasattr(graph, 'x') and hasattr(graph, 'y'):
                features = graph.x.numpy()
                labels = graph.y.numpy()
                
                all_features.append(features)
                all_labels.append(labels)
                
                print(f"  {gf}: {features.shape[0]} samples, {features.shape[1]} features")
        
        except Exception as e:
            print(f"  Error loading {gf}: {e}")
    
    if not all_features:
        print("❌ No features loaded")
        return
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"\n📊 Combined dataset:")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive samples: {y.sum()} ({y.sum()/len(y):.1%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train simple model for feature importance
    print(f"\n🏋️ Training Random Forest for feature importance...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📈 Model Performance:")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    
    # Feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n🔝 Top 15 Most Important Features:")
    
    # Create feature names (estimate)
    feature_names = []
    for i in range(X.shape[1]):
        if i < 20:
            residue_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
            feature_names.append(f"Residue_{residue_types[i]}" if i < len(residue_types) else f"Residue_{i}")
        elif i == 20:
            feature_names.append("Partial_Charge")
        elif i == 21:
            feature_names.append("VDW_Radius")
        elif i == 22:
            feature_names.append("HB_Donor")
        elif i == 23:
            feature_names.append("HB_Acceptor")
        elif i == 24:
            feature_names.append("Hydrophobicity")
        elif i == 25:
            feature_names.append("Charge")
        elif i == 26:
            feature_names.append("Size")
        elif i == 27:
            feature_names.append("Position_X")
        elif i == 28:
            feature_names.append("Position_Y")
        elif i == 29:
            feature_names.append("Position_Z")
        else:
            feature_names.append(f"Feature_{i}")
    
    for i in range(min(15, len(indices))):
        idx = indices[i]
        print(f"   {i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:.6f}")
    
    # Save results
    output_dir = "./experiments/results/phase3/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model_performance': {
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'n_samples': int(X.shape[0])
        },
        'feature_importance': {
            'feature_names': feature_names,
            'importance_scores': importances.tolist(),
            'top_features': []
        }
    }
    
    # Add top features
    for i in range(min(20, len(indices))):
        idx = indices[i]
        results['feature_importance']['top_features'].append({
            'rank': i+1,
            'feature_index': int(idx),
            'feature_name': feature_names[idx],
            'importance': float(importances[idx])
        })
    
    with open(f"{output_dir}/feature_importance_rf.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Feature importance saved to {output_dir}/feature_importance_rf.json")
    
    # Create visualization
    create_importance_plot(feature_names, importances, indices, output_dir)
    
    return results

def create_importance_plot(feature_names, importances, indices, output_dir):
    """Create feature importance visualization"""
    import matplotlib.pyplot as plt
    
    # Top 15 features
    top_n = 15
    top_indices = indices[:top_n]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = [importances[i] for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.barh(range(len(top_names)), top_scores, color='darkorange', alpha=0.7)
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        plt.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=9)
    
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance Score')
    plt.title('Top 15 Important Features for Pocket Detection (Random Forest)')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/feature_importance_plot.pdf", bbox_inches='tight')
    
    print(f"📊 Feature importance plot saved")
    plt.close()

if __name__ == "__main__":
    analyze_feature_importance()
