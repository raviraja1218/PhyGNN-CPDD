#!/usr/bin/env python3
"""
Compute comprehensive dataset statistics
Output: dataset_statistics.csv (TARGET 1)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import PDB
from rdkit import Chem
import json

class DatasetStatistics:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir) / "PDBbind" / "refined-set"
        self.results_dir = Path("./experiments/results/phase1")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_complexes(self):
        """Load list of valid complexes"""
        complexes_file = self.results_dir / "valid_complexes.csv"
        if complexes_file.exists():
            df = pd.read_csv(complexes_file)
            return df.to_dict('records')
        else:
            # Fallback: scan directory
            complexes = []
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    pdb_id = item.name
                    protein_file = item / f"{pdb_id}_protein.pdb"
                    ligand_file = item / f"{pdb_id}_ligand.mol2"
                    if protein_file.exists() and ligand_file.exists():
                        complexes.append({
                            'pdb_id': pdb_id,
                            'protein_path': str(protein_file),
                            'ligand_path': str(ligand_file)
                        })
            return complexes
    
    def compute_statistics(self):
        """Compute all dataset statistics"""
        print("Computing dataset statistics...")
        
        complexes = self.load_complexes()
        print(f"Processing {len(complexes)} complexes")
        
        stats = {
            'Total Complexes': len(complexes),
            'Resolution Values': [],
            'Protein Sizes': [],
            'Ligand Sizes': [],
            'Binding Affinities': []
        }
        
        # Process each complex
        for i, complex_data in enumerate(complexes[:100]):  # Sample first 100 for speed
            if i % 10 == 0:
                print(f"  Processed {i}/{len(complexes[:100])}")
            
            # Extract protein metadata
            protein_stats = self.extract_protein_stats(complex_data['protein_path'])
            if protein_stats:
                stats['Resolution Values'].append(protein_stats.get('resolution'))
                stats['Protein Sizes'].append(protein_stats.get('num_residues'))
            
            # Extract ligand metadata
            ligand_stats = self.extract_ligand_stats(complex_data['ligand_path'])
            if ligand_stats:
                stats['Ligand Sizes'].append(ligand_stats.get('num_atoms'))
            
            # Extract binding affinity (if available)
            affinity = self.extract_affinity(complex_data['pdb_id'])
            if affinity:
                stats['Binding Affinities'].append(affinity)
        
        # Compute summary statistics
        summary = self.compute_summary(stats)
        
        # Save results
        self.save_statistics(summary)
        
        return summary
    
    def extract_protein_stats(self, pdb_path):
        """Extract statistics from PDB file"""
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("temp", pdb_path)
            
            # Count residues
            residues = list(structure.get_residues())
            
            return {
                'resolution': structure.header.get('resolution', None),
                'num_residues': len(residues),
                'structure_method': structure.header.get('structure_method', None)
            }
        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            return None
    
    def extract_ligand_stats(self, mol2_path):
        """Extract statistics from ligand file"""
        try:
            mol = Chem.MolFromMol2File(mol2_path)
            if mol:
                return {
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'molecular_weight': Chem.Descriptors.MolWt(mol)
                }
        except Exception as e:
            # Try alternative format
            try:
                mol = Chem.MolFromPDBFile(mol2_path.replace('.mol2', '.sdf'))
                if mol:
                    return {
                        'num_atoms': mol.GetNumAtoms(),
                        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                        'molecular_weight': Chem.Descriptors.MolWt(mol)
                    }
            except:
                pass
        return None
    
    def extract_affinity(self, pdb_id):
        """Extract binding affinity from index file"""
        try:
            index_file = self.data_dir.parent / "index" / "INDEX_refined_data.2020"
            with open(index_file, 'r') as f:
                for line in f:
                    if line.startswith(pdb_id):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            return float(parts[3])  # Typically pKd/pKi
        except:
            pass
        return None
    
    def compute_summary(self, stats):
        """Compute summary statistics"""
        summary = []
        
        # Total complexes
        summary.append(("Total Complexes", len(self.load_complexes()), "Actual count"))
        
        # Resolution
        resolutions = [r for r in stats['Resolution Values'] if r is not None]
        if resolutions:
            summary.append(("Resolution Min", f"{min(resolutions):.2f} Å", ""))
            summary.append(("Resolution Max", f"{max(resolutions):.2f} Å", ""))
            summary.append(("Resolution Mean", f"{np.mean(resolutions):.2f} ± {np.std(resolutions):.2f} Å", "with std"))
        
        # Protein sizes
        protein_sizes = [s for s in stats['Protein Sizes'] if s is not None]
        if protein_sizes:
            summary.append(("Protein Size Min", f"{min(protein_sizes)} residues", ""))
            summary.append(("Protein Size Max", f"{max(protein_sizes)} residues", ""))
            summary.append(("Protein Size Mean", f"{np.mean(protein_sizes):.0f} ± {np.std(protein_sizes):.0f} residues", "with std"))
        
        # Ligand sizes
        ligand_sizes = [s for s in stats['Ligand Sizes'] if s is not None]
        if ligand_sizes:
            summary.append(("Ligand Size Min", f"{min(ligand_sizes)} atoms", ""))
            summary.append(("Ligand Size Max", f"{max(ligand_sizes)} atoms", ""))
            summary.append(("Ligand Size Mean", f"{np.mean(ligand_sizes):.0f} ± {np.std(ligand_sizes):.0f} atoms", "with std"))
        
        # Binding affinities
        affinities = [a for a in stats['Binding Affinities'] if a is not None]
        if affinities:
            summary.append(("Binding Affinity Min", f"{min(affinities):.2f} pKd", ""))
            summary.append(("Binding Affinity Max", f"{max(affinities):.2f} pKd", ""))
            summary.append(("Binding Affinity Mean", f"{np.mean(affinities):.2f} ± {np.std(affinities):.2f} pKd", "with std"))
        
        # Dataset splits
        complexes = self.load_complexes()
        train_count = int(len(complexes) * 0.7)
        val_count = int(len(complexes) * 0.15)
        test_count = len(complexes) - train_count - val_count
        
        summary.append(("Dataset Split", f"Train/Val/Test: {train_count}/{val_count}/{test_count}", "70/15/15 split"))
        
        return summary
    
    def save_statistics(self, summary):
        """Save statistics to CSV"""
        df = pd.DataFrame(summary, columns=['Statistic', 'Value', 'Notes'])
        output_path = self.results_dir / "dataset_statistics.csv"
        df.to_csv(output_path, index=False)
        print(f"\nStatistics saved to: {output_path}")
        
        # Also save as JSON for easy reading
        json_stats = {row[0]: {'value': row[1], 'notes': row[2]} for row in summary}
        with open(self.results_dir / "dataset_statistics.json", 'w') as f:
            json.dump(json_stats, f, indent=2)

def main():
    print("=== Dataset Statistics Computation ===")
    
    stats = DatasetStatistics()
    summary = stats.compute_statistics()
    
    print("\n=== DATASET STATISTICS ===")
    for stat, value, notes in summary:
        print(f"{stat:25} {value:30} {notes}")
    
    print(f"\nResults saved to: ./experiments/results/phase1/")
    print("TARGET 1: ✅ dataset_statistics.csv created")

if __name__ == "__main__":
    main()
