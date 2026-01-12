#!/usr/bin/env python
"""
Test GPU and PyTorch Geometric setup.
"""
import torch
import torch_geometric
import numpy as np
import pandas as pd

print("=" * 70)
print("PhyGNN-CPDD GPU Setup Verification")
print("=" * 70)

# 1. PyTorch and CUDA info
print("\n1. PyTorch & CUDA Information:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU Device: {gpu_name}")
    print(f"   GPU Memory: {gpu_memory:.2f} GB")
    print(f"   Current device: {torch.cuda.current_device()}")
else:
    device = torch.device('cpu')
    print("   ❌ CUDA not available - using CPU")

# 2. Test tensor operations on GPU
print("\n2. GPU Tensor Operations Test:")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.matmul(x, y)
print(f"   Matrix multiplication on {device}: {z.shape}")
print(f"   Device of result: {z.device}")

# 3. PyTorch Geometric test
print("\n3. PyTorch Geometric Test:")
try:
    from torch_geometric.data import Data
    
    # Create a simple graph
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long, device=device)
    x = torch.randn(3, 16, device=device)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"   Created graph data on {device}")
    print(f"   Graph nodes: {data.num_nodes}, edges: {data.num_edges}")
    
    # Test GNN layer
    from torch_geometric.nn import GCNConv
    conv = GCNConv(16, 32).to(device)
    out = conv(data.x, data.edge_index)
    print(f"   GCNConv output shape on {device}: {out.shape}")
    
    print("   ✅ PyTorch Geometric working correctly")
except Exception as e:
    print(f"   ❌ PyTorch Geometric error: {e}")

# 4. Memory test
print("\n4. GPU Memory Test:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e6
    reserved = torch.cuda.memory_reserved(0) / 1e6
    print(f"   Allocated: {allocated:.2f} MB")
    print(f"   Reserved: {reserved:.2f} MB")
    
    # Test large allocation
    try:
        large_tensor = torch.randn(5000, 5000, device=device)
        print(f"   Large tensor (5000x5000) allocated successfully")
        del large_tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"   ❌ Large allocation failed: {e}")

# 5. Dataset access test
print("\n5. Dataset Access Test:")
try:
    import os
    data_path = "./data/PDBbind"
    if os.path.exists(data_path):
        print(f"   ✅ Dataset found at: {data_path}")
        # Count complexes
        import subprocess
        result = subprocess.run(['find', data_path, '-maxdepth', '2', '-type', 'd'], 
                              capture_output=True, text=True)
        dirs = [d for d in result.stdout.strip().split('\n') if d]
        print(f"   Found {len(dirs)} directories")
    else:
        print("   ❌ Dataset not found")
except Exception as e:
    print(f"   Dataset access error: {e}")

print("\n" + "=" * 70)
print("GPU Setup Complete!")
print("=" * 70)

# Recommendations
print("\n🎯 Next Steps:")
print("1. Run: python test_gpu_setup.py")
print("2. Open: jupyter notebook notebooks/01_data_exploration.ipynb")
print("3. Start implementing Phase 1 from execution plan")
