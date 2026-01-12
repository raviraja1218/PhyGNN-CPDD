#!/usr/bin/env python
"""
Test script to verify PhyGNN-CPDD setup.
"""
import sys
import os

# Add src to path
sys.path.append('src')

print("=" * 60)
print("PhyGNN-CPDD Setup Verification")
print("=" * 60)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import torch
    import torch_geometric
    import numpy as np
    import pandas as pd
    import matplotlib
    print("✅ Basic imports successful")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test 2: Biopython and RDKit
print("\n2. Testing bioinformatics libraries...")
try:
    import Bio
    from Bio import PDB
    import rdkit
    from rdkit import Chem
    print("✅ Bioinformatics libraries successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test 3: Data loader
print("\n3. Testing data loader...")
try:
    from data_processing.pdb_loader import PDBbindLoader
    loader = PDBbindLoader()
    print("✅ Data loader initialized")
    
    # Try to list complexes
    complexes = loader.list_complexes()
    print(f"   Found {len(complexes)} complexes")
    
    # Load stats
    stats = loader.get_dataset_stats()
    print(f"   Sample PDBs: {stats.get('sample_pdbs', [])[:3]}")
    
except Exception as e:
    print(f"❌ Data loader failed: {e}")

# Test 4: Dataset access
print("\n4. Testing dataset access...")
try:
    import os
    data_path = "./data/PDBbind"
    if os.path.exists(data_path):
        print(f"✅ Dataset found at {data_path}")
        # Count directories
        import subprocess
        result = subprocess.run(
            ['find', data_path, '-maxdepth', '2', '-type', 'd'],
            capture_output=True, text=True
        )
        dirs = result.stdout.strip().split('\n')
        print(f"   Found {len([d for d in dirs if d])} directories")
    else:
        print("❌ Dataset not found")
except Exception as e:
    print(f"❌ Dataset access failed: {e}")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)

# Recommendations
print("\nNext steps:")
print("1. Run: python test_setup.py")
print("2. Open: notebooks/01_data_exploration.ipynb")
print("3. Check: experiments/configs/default.yaml")
