#!/usr/bin/env python3
"""
Validate PDBbind Dataset
Check all files exist and are readable
"""

import os
import sys
from pathlib import Path
from Bio import PDB
import pandas as pd

def count_complexes(data_dir):
    """Count total protein-ligand complexes"""
    base_path = Path(data_dir) / "PDBbind" / "refined-set"
    
    complexes = []
    issues = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            pdb_id = item.name
            protein_file = item / f"{pdb_id}_protein.pdb"
            ligand_file = item / f"{pdb_id}_ligand.mol2"
            
            if protein_file.exists() and ligand_file.exists():
                complexes.append({
                    'pdb_id': pdb_id,
                    'protein_path': str(protein_file),
                    'ligand_path': str(ligand_file),
                    'has_protein': protein_file.exists(),
                    'has_ligand': ligand_file.exists()
                })
            else:
                issues.append({
                    'pdb_id': pdb_id,
                    'missing_protein': not protein_file.exists(),
                    'missing_ligand': not ligand_file.exists()
                })
    
    return complexes, issues

def extract_metadata(pdb_path):
    """Extract resolution and metadata from PDB file"""
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("temp", pdb_path)
        
        metadata = {
            'resolution': structure.header.get('resolution', None),
            'structure_method': structure.header.get('structure_method', None),
            'deposition_date': structure.header.get('deposition_date', None),
            'r_free': structure.header.get('r_free', None),
            'r_work': structure.header.get('r_work', None),
        }
        
        # Count residues
        residues = list(structure.get_residues())
        metadata['num_residues'] = len(residues)
        
        return metadata
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=== PDBbind Dataset Validation ===")
    
    data_dir = "./data"
    complexes, issues = count_complexes(data_dir)
    
    print(f"\nTotal complexes found: {len(complexes)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nProblematic complexes:")
        for issue in issues[:5]:  # Show first 5
            print(f"  {issue['pdb_id']}: protein={issue['missing_protein']}, ligand={issue['missing_ligand']}")
    
    # Sample first 5 complexes for metadata
    print("\n\nSampling first 5 complexes:")
    for complex_data in complexes[:5]:
        metadata = extract_metadata(complex_data['protein_path'])
        print(f"\n{complex_data['pdb_id']}:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = "./experiments/results/phase1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save list of valid complexes
    df = pd.DataFrame(complexes)
    df.to_csv(f"{output_dir}/valid_complexes.csv", index=False)
    
    print(f"\n\nResults saved to: {output_dir}/valid_complexes.csv")
    
    return len(complexes)

if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 1000 else 1)  # Exit with error if less than 1000 complexes
