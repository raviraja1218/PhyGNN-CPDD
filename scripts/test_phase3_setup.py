#!/usr/bin/env python3
"""
Test Phase 3 setup
"""
import os
import torch

print("Testing Phase 3 setup...")

# 1. Test model loading
print("\n1. Testing model loading...")
try:
    from src.models.hamiltonian_gnn_ablation_fixed import HamiltonianGNN
    
    model_path = './experiments/results/phase2b/week2/training_fixed/hamgnn_best.pt'
    model_state = torch.load(model_path, weights_only=True, map_location='cpu')
    
    model = HamiltonianGNN(input_dim=35, hidden_dim=128, lambda_physics=0.0001)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    
    print(f"   ✅ Model loaded")
    print(f"   Missing keys (physics, OK): {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    
except Exception as e:
    print(f"   ❌ Model load failed: {e}")

# 2. Test data availability
print("\n2. Testing data availability...")

# Check multiple possible locations
possible_locations = [
    "./data/processed/physics_graphs/train/",
    "./data/processed/phase2c_final_300_converted/", 
    "./data/processed/graphs_working/train/",
    "./data/processed/graphs_simple_enhanced/"
]

found_files = []
for loc in possible_locations:
    if os.path.exists(loc):
        files = [f for f in os.listdir(loc) if f.endswith('.pt')]
        if files:
            print(f"   ✅ Found {len(files)} .pt files in {loc}")
            found_files.extend([os.path.join(loc, f) for f in files[:3]])
        else:
            print(f"   ⚠️ No .pt files in {loc}")

if found_files:
    print(f"   Sample files: {found_files[:3]}")
else:
    print("   ❌ No graph files found!")
    
    # Let's check raw PDB files instead
    pdb_dir = "./data/PDBbind/refined-set"
    if os.path.exists(pdb_dir):
        pdb_folders = os.listdir(pdb_dir)[:5]
        print(f"   Raw PDB folders exist: {pdb_folders}")

# 3. Test captum installation
print("\n3. Testing Captum installation...")
try:
    import captum
    print(f"   ✅ Captum version: {captum.__version__}")
except ImportError as e:
    print(f"   ❌ Captum not installed: {e}")

print("\n" + "="*50)
print("SETUP TEST COMPLETE")
