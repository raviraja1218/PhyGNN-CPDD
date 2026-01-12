#!/usr/bin/env python3
"""
Extract resolution from PDBbind files (alternative method)
"""

import os
import re
from pathlib import Path
import pandas as pd

def extract_resolution_from_file(pdb_file):
    """Extract resolution from PDB file using regex patterns"""
    try:
        with open(pdb_file, 'r') as f:
            content = f.read()
            
        # Pattern 1: Standard PDB REMARK 2 lines
        # REMARK   2 RESOLUTION.    2.20 ANGSTROMS.
        pattern1 = r'REMARK\s+2\s+RESOLUTION\.\s+([\d\.]+)\s+ANGSTROMS?'
        match1 = re.search(pattern1, content, re.IGNORECASE)
        if match1:
            return float(match1.group(1))
        
        # Pattern 2: Alternative format
        # REMARK   2   RESOLUTION RANGE HIGH (ANGSTROMS) : 2.20
        pattern2 = r'RESOLUTION.*?([\d\.]+)\s*ANGSTROMS?'
        match2 = re.search(pattern2, content, re.IGNORECASE)
        if match2:
            return float(match2.group(1))
        
        # Pattern 3: Look in other remark lines
        for line in content.split('\n'):
            if 'RESOLUTION' in line.upper() and 'ANGSTROM' in line.upper():
                numbers = re.findall(r'[\d\.]+', line)
                if numbers:
                    return float(numbers[0])
        
        return None
    except:
        return None

def main():
    print("=== Extracting Resolution Data ===")
    
    data_dir = Path("./data/PDBbind/refined-set")
    results = []
    
    # Process first 100 proteins for speed
    count = 0
    for pdb_dir in data_dir.iterdir():
        if pdb_dir.is_dir() and count < 100:
            pdb_id = pdb_dir.name
            protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
            
            if protein_file.exists():
                resolution = extract_resolution_from_file(protein_file)
                if resolution:
                    results.append({'pdb_id': pdb_id, 'resolution': resolution})
                    print(f"{pdb_id}: {resolution} Å")
                count += 1
    
    # Save results
    output_dir = Path("./experiments/results/phase1")
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "resolution_data.csv", index=False)
    
    if len(results) > 0:
        print(f"\n✅ Extracted resolution for {len(results)} proteins")
        print(f"Mean resolution: {df['resolution'].mean():.2f} Å")
        print(f"Min resolution: {df['resolution'].min():.2f} Å")
        print(f"Max resolution: {df['resolution'].max():.2f} Å")
    else:
        print("❌ No resolution data found")
    
    return df

if __name__ == "__main__":
    main()
