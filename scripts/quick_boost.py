"""
Quick boost for F1 score
"""
import torch
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score

print("Quick Boost Analysis")
print("="*60)

# Load a sample graph to analyze
graph_dir = "./data/processed/graphs_enhanced/train"
graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')][:10]

print(f"Analyzing {len(graph_files)} graphs...")

# Calculate statistics
all_features = []
all_labels = []

for graph_file in graph_files:
    graph_path = os.path.join(graph_dir, graph_file)
    graph = torch.load(graph_path, weights_only=False)
    
    features = graph.x.numpy()
    labels = graph.y.numpy()
    
    all_features.append(features)
    all_labels.append(labels)

# Combine all
all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)

print(f"\nDataset Statistics:")
print(f"Total samples: {len(all_labels)}")
print(f"Positive samples: {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
print(f"Negative samples: {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")

# Feature analysis
print(f"\nFeature Analysis:")
print(f"Feature dimension: {all_features.shape[1]}")
print(f"Feature mean: {all_features.mean():.3f} ± {all_features.std():.3f}")
print(f"Feature range: [{all_features.min():.3f}, {all_features.max():.3f}]")

# Check for feature differences between classes
pos_features = all_features[all_labels == 1]
neg_features = all_features[all_labels == 0]

if len(pos_features) > 0:
    print(f"\nClass-wise feature differences:")
    for i in range(min(5, all_features.shape[1])):  # First 5 features
        pos_mean = pos_features[:, i].mean()
        neg_mean = neg_features[:, i].mean()
        diff = pos_mean - neg_mean
        print(f"  Feature {i}: Pos={pos_mean:.3f}, Neg={neg_mean:.3f}, Diff={diff:.3f}")

# Simple heuristic: can we separate classes with a simple rule?
print(f"\nSimple Heuristic Test:")
# Use first feature as example (usually residue type)
if all_features.shape[1] > 20:  # If we have one-hot encoding
    # Look at hydrophobic residues (ALA, VAL, LEU, ILE, PHE, etc.)
    hydrophobic_indices = [0, 9, 10, 13, 17]  # ALA, ILE, LEU, PHE, TRP
    hydrophobic_score = all_features[:, hydrophobic_indices].sum(axis=1)
    
    # Simple threshold
    threshold = hydrophobic_score.mean()
    preds = (hydrophobic_score > threshold).astype(float)
    
    f1 = f1_score(all_labels, preds, zero_division=0)
    print(f"  Hydrophobicity heuristic F1: {f1:.4f}")

print(f"\nRecommendations:")
print("1. If feature differences are small → Need better features")
print("2. If class imbalance severe (<5%) → Need oversampling or stronger loss")
print("3. If simple heuristics work (>0.10 F1) → Model should learn")
print("4. Check if graphs are too large/small → May need pooling")
