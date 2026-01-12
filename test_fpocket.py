#!/usr/bin/env python3
"""
Test FPOCKET on a single protein
"""

import subprocess
import os
from pathlib import Path

def test_fpocket():
    # Find first protein in dataset
    data_dir = Path("./data/PDBbind/refined-set")
    
    for pdb_dir in data_dir.iterdir():
        if pdb_dir.is_dir():
            pdb_id = pdb_dir.name
            protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
            
            if protein_file.exists():
                print(f"Testing FPOCKET on {pdb_id}...")
                
                # Try direct fpocket command
                try:
                    cmd = ["fpocket", "-f", str(protein_file)]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print("✅ FPOCKET works!")
                        print(f"Output: {result.stdout[:200]}...")
                        return True
                    else:
                        print(f"❌ FPOCKET failed: {result.stderr[:100]}")
                except FileNotFoundError:
                    print("❌ fpocket command not found")
                
                # Try Docker version
                try:
                    cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/data", 
                           "discngine/fpocket", "fpocket", "-f", f"/data/{protein_file}"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("✅ Docker FPOCKET works!")
                        # Create alias for future use
                        with open("fpocket_docker.sh", "w") as f:
                            f.write('''#!/bin/bash
docker run --rm -v $(pwd):/data discngine/fpocket fpocket "$@"
''')
                        os.chmod("fpocket_docker.sh", 0o755)
                        print("Created fpocket_docker.sh wrapper")
                        return True
                except Exception as e:
                    print(f"❌ Docker also failed: {e}")
                
                break  # Only test first protein
    
    return False

if __name__ == "__main__":
    success = test_fpocket()
    if not success:
        print("\n⚠️ FPOCKET setup failed. Options:")
        print("1. Install manually from: https://github.com/Discngine/fpocket")
        print("2. Use alternative pocket detector (we can implement our own)")
        print("3. Skip FPOCKET and use literature value (F1=0.52)")
