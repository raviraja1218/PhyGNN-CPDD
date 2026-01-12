#!/usr/bin/env python3
"""
Python-only baseline for pocket detection
Uses geometric and chemical features
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import sys
import os

# Try to import BioPython, but provide fallback
try:
    from Bio import PDB
    from Bio.PDB import NeighborSearch
    HAS_BIOPYTHON = True
except ImportError:
    print("⚠️ BioPython not available. Using simplified version.")
    HAS_BIOPYTHON = False

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    print("⚠️ RDKit not available. Using simplified version.")
    HAS_RDKIT = False

class PythonBaseline:
    def __init__(self):
        self.data_dir = Path("./data/PDBbind/refined-set")
        self.results_dir = Path("./experiments/results/phase1")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        self.pocket_cutoff = 4.0  # Ångstroms for ground truth
        self.surface_cutoff = 6.0  # Ångstroms for predicted pockets
        
    def load_protein_coords_simple(self, pdb_file):
        """Simple PDB parser without BioPython"""
        coords = []
        residues = []
        
        try:
            with open(pdb_file, 'r') as f:
                current_residue = None
                residue_coords = []
                
                for line in f:
                    if line.startswith('ATOM'):
                        # Parse ATOM line (simplified)
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21]
                        residue_num = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        residue_key = f"{chain_id}:{residue_num}:{residue_name}"
                        
                        if current_residue != residue_key:
                            if current_residue and residue_coords:
                                coords.append(np.mean(residue_coords, axis=0))
                                residues.append(current_residue)
                            current_residue = residue_key
                            residue_coords = []
                        
                        residue_coords.append([x, y, z])
                
                # Add last residue
                if current_residue and residue_coords:
                    coords.append(np.mean(residue_coords, axis=0))
                    residues.append(current_residue)
                    
            return np.array(coords), residues
            
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            return None, None
    
    def load_ligand_coords_simple(self, mol2_file):
        """Simple MOL2 parser without RDKit"""
        coords = []
        in_atom_section = False
        
        try:
            with open(mol2_file, 'r') as f:
                for line in f:
                    if line.startswith('@<TRIPOS>ATOM'):
                        in_atom_section = True
                        continue
                    elif line.startswith('@<TRIPOS>'):
                        in_atom_section = False
                    
                    if in_atom_section and line.strip():
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                                coords.append([x, y, z])
                            except:
                                pass
        except:
            pass
        
        return np.array(coords) if coords else None
    
    def detect_pockets_geometric(self, protein_coords, protein_residues):
        """Simple geometric pocket detection"""
        if protein_coords is None or len(protein_coords) == 0:
            return []
        
        # Calculate pairwise distances
        n_residues = len(protein_coords)
        if n_residues > 1000:  # For speed, sample if too large
            indices = np.random.choice(n_residues, min(1000, n_residues), replace=False)
            sampled_coords = protein_coords[indices]
            sampled_residues = [protein_residues[i] for i in indices]
        else:
            sampled_coords = protein_coords
            sampled_residues = protein_residues
        
        # Simple algorithm: Find concave regions
        # For each residue, find neighbors within radius
        predicted = []
        
        for i, (coord, residue) in enumerate(zip(sampled_coords, sampled_residues)):
            # Calculate distances to all other residues
            distances = np.linalg.norm(sampled_coords - coord, axis=1)
            
            # Count neighbors within 8Å
            neighbor_count = np.sum(distances < 8.0)
            
            # If few neighbors, might be on surface
            if neighbor_count < 15:  # Empirical threshold
                # Check if it's in a concave region
                # Find closest neighbors
                close_indices = np.where(distances < 12.0)[0]
                if len(close_indices) > 5:
                    # Calculate local density
                    local_coords = sampled_coords[close_indices]
                    centroid = np.mean(local_coords, axis=0)
                    
                    # If residue is significantly inside the centroid, it's in a pocket
                    if np.linalg.norm(coord - centroid) < 5.0:
                        predicted.append(residue)
        
        return predicted
    
    def get_ground_truth_simple(self, protein_coords, protein_residues, ligand_coords):
        """Get ground truth pockets"""
        if ligand_coords is None or len(ligand_coords) == 0:
            return []
        
        ground_truth = []
        
        for i, (coord, residue) in enumerate(zip(protein_coords, protein_residues)):
            # Calculate minimum distance to any ligand atom
            distances = np.linalg.norm(ligand_coords - coord, axis=1)
            min_dist = np.min(distances)
            
            if min_dist <= self.pocket_cutoff:
                ground_truth.append(residue)
        
        return ground_truth
    
    def calculate_metrics(self, predicted, ground_truth):
        """Calculate precision, recall, F1"""
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1, precision, recall, tp, fp, fn
    
    def process_protein(self, pdb_id):
        """Process a single protein"""
        protein_file = self.data_dir / pdb_id / f"{pdb_id}_protein.pdb"
        ligand_file = self.data_dir / pdb_id / f"{pdb_id}_ligand.mol2"
        
        if not protein_file.exists() or not ligand_file.exists():
            return None
        
        # Load data
        protein_coords, protein_residues = self.load_protein_coords_simple(protein_file)
        ligand_coords = self.load_ligand_coords_simple(ligand_file)
        
        if protein_coords is None or ligand_coords is None:
            return None
        
        # Detect pockets
        predicted = self.detect_pockets_geometric(protein_coords, protein_residues)
        
        # Get ground truth
        ground_truth = self.get_ground_truth_simple(protein_coords, protein_residues, ligand_coords)
        
        if not ground_truth:
            return None
        
        # Calculate metrics
        f1, precision, recall, tp, fp, fn = self.calculate_metrics(predicted, ground_truth)
        
        return {
            'pdb_id': pdb_id,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_predicted': len(predicted),
            'num_ground_truth': len(ground_truth),
            'protein_size': len(protein_residues)
        }
    
    def run_baseline(self, num_proteins=50):
        """Run baseline evaluation"""
        print("=== Python Geometric Baseline ===")
        print(f"Processing up to {num_proteins} proteins...")
        
        # Get list of proteins
        protein_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                pdb_id = item.name
                protein_file = item / f"{pdb_id}_protein.pdb"
                ligand_file = item / f"{pdb_id}_ligand.mol2"
                if protein_file.exists() and ligand_file.exists():
                    protein_dirs.append(pdb_id)
        
        # Process proteins
        results = []
        processed = 0
        
        for pdb_id in protein_dirs[:num_proteins]:
            processed += 1
            print(f"[{processed}/{min(num_proteins, len(protein_dirs))}] {pdb_id}", end="")
            
            result = self.process_protein(pdb_id)
            
            if result:
                results.append(result)
                print(f" - F1: {result['f1']:.3f}")
            else:
                print(" - Skipped")
        
        if not results:
            print("❌ No results generated")
            return None
        
        # Save results
        df = pd.DataFrame(results)
        output_file = self.results_dir / "python_baseline_results.csv"
        df.to_csv(output_file, index=False)
        
        # Calculate statistics
        avg_f1 = df['f1'].mean()
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        
        print(f"\n=== RESULTS ===")
        print(f"Processed: {len(results)} proteins")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"F1 Range: {df['f1'].min():.3f} - {df['f1'].max():.3f}")
        
        # Compare with literature
        print(f"\n=== COMPARISON WITH LITERATURE ===")
        print(f"Our F1: {avg_f1:.3f}")
        print(f"FPOCKET literature F1: 0.52")
        print(f"Difference: {avg_f1 - 0.52:+.3f}")
        
        if avg_f1 >= 0.40:
            print("✅ Baseline is reasonable (≥0.40 F1)")
        else:
            print("⚠️ Baseline is lower than expected")
        
        # Save metrics for paper
        metrics = {
            'baseline_method': 'python_geometric',
            'overall_f1': float(avg_f1),
            'overall_precision': float(avg_precision),
            'overall_recall': float(avg_recall),
            'num_proteins_evaluated': len(results),
            'f1_std': float(df['f1'].std()),
            'note': 'Python-only geometric baseline using distance-based heuristics'
        }
        
        metrics_file = self.results_dir / "baseline_performance.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create LaTeX table
        self.create_latex_table(metrics, df)
        
        return metrics
    
    def create_latex_table(self, metrics, df):
        """Create LaTeX table for paper"""
        latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Python Geometric Baseline Performance}}
\\label{{tab:python_baseline}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Literature (FPOCKET)}} & \\textbf{{Difference}} \\\\
\\midrule
F1 Score & {metrics['overall_f1']:.3f} ± {metrics['f1_std']:.3f} & 0.520 ± 0.040 & {metrics['overall_f1'] - 0.52:+.3f} \\\\
Precision & {metrics['overall_precision']:.3f} & 0.480 & {metrics['overall_precision'] - 0.48:+.3f} \\\\
Recall & {metrics['overall_recall']:.3f} & 0.570 & {metrics['overall_recall'] - 0.57:+.3f} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\small Evaluated on {metrics['num_proteins_evaluated']} proteins from PDBbind}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        
        latex_file = self.results_dir / "baseline_comparison.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        
        print(f"\n✅ LaTeX table saved to: {latex_file}")

def main():
    baseline = PythonBaseline()
    metrics = baseline.run_baseline(num_proteins=50)
    
    if metrics:
        print(f"\n✅ Python baseline completed successfully!")
        print(f"Results saved to: ./experiments/results/phase1/")
        print("- python_baseline_results.csv")
        print("- baseline_performance.json")
        print("- baseline_comparison.tex")
    else:
        print("❌ Baseline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
