#!/usr/bin/env python3
"""
Find and use whatever data is available
"""
import os
import torch

def find_graph_files():
    """Find all available graph files"""
    all_files = []
    
    # Search recursively
    for root, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith('.pt') and 'graph' in file:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    
    return all_files

# Find files
files = find_graph_files()
print(f"Found {len(files)} graph files total")

if files:
    # Take first 10 for testing
    test_files = files[:10]
    print(f"\nFirst 10 files:")
    for f in test_files:
        print(f"  {f}")
    
    # Test loading one
    print(f"\nTesting load of {test_files[0]}...")
    try:
        graph = torch.load(test_files[0])
        print(f"  ✅ Loaded successfully")
        print(f"  Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
        print(f"  Features: {graph.x.shape}")
        print(f"  Has y: {hasattr(graph, 'y')}")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
else:
    print("❌ No graph files found!")
    
    # Maybe we need to process raw PDB files
    print("\nChecking for raw PDB files...")
    pdb_count = 0
    for root, dirs, files in os.walk("./data/PDBbind"):
        for file in files:
            if file.endswith('.pdb') and 'protein' in file:
                pdb_count += 1
    
    print(f"Found {pdb_count} raw PDB files")
    
    if pdb_count > 0:
        print("We need to process PDB files into graphs first!")
        print("Run: python3 ./src/models/physics_enhanced_builder.py")
