#!/usr/bin/env python3
"""
Run FPOCKET baseline on all proteins
Compute performance metrics (TARGET 2)
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class FPocketBaseline:
    def __init__(self):
        self.data_dir = Path("./data/PDBbind/refined-set")
        self.results_dir = Path("./experiments/results/phase1")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load splits
        splits_dir = self.results_dir / "splits"
        self.train_ids = self.load_ids(splits_dir / "train_ids.txt")
        self.val_ids = self.load_ids(splits_dir / "val_ids.txt")
        self.test_ids = self.load_ids(splits_dir / "test_ids.txt")
        
        # Ground truth parameters
        self.pocket_cutoff = 4.0  # Ångstroms
        self.min_pocket_residues = 5  # Minimum residues to consider a pocket
        
    def load_ids(self, filepath):
        """Load list of IDs from file"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    def run_fpocket(self, pdb_id):
        """Run FPOCKET on a single protein"""
        protein_file = self.data_dir / pdb_id / f"{pdb_id}_protein.pdb"
        output_dir = self.results_dir / "fpocket_output" / pdb_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Command to run FPOCKET
        cmd = [
            "fpocket",
            "-f", str(protein_file),
            "-o", str(output_dir)
        ]
        
        try:
            # Run FPOCKET
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {pdb_id}: FPOCKET completed")
                return True, str(output_dir)
            else:
                print(f"  ❌ {pdb_id}: FPOCKET failed - {result.stderr[:100]}")
                return False, result.stderr
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {pdb_id}: FPOCKET timeout")
            return False, "Timeout"
        except Exception as e:
            print(f"  ❌ {pdb_id}: Error - {str(e)}")
            return False, str(e)
    
    def parse_fpocket_output(self, pdb_id, output_dir):
        """Parse FPOCKET output to get predicted pockets"""
        pockets_file = Path(output_dir) / f"{pdb_id}_protein_out" / f"{pdb_id}_protein_pockets.pqr"
        
        if not pockets_file.exists():
            return []
        
        predicted_pockets = []
        current_pocket = None
        
        with open(pockets_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Extract atom information
                    atom_serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21]
                    residue_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # Pocket number is in the occupancy column (54-60)
                    pocket_num = int(float(line[54:60].strip()))
                    
                    if pocket_num > 0:
                        if current_pocket != pocket_num:
                            predicted_pockets.append({
                                'pocket_id': pocket_num,
                                'residues': [],
                                'atoms': [],
                                'center': None
                            })
                            current_pocket = pocket_num
                        
                        predicted_pockets[-1]['residues'].append(f"{chain_id}:{residue_num}:{residue_name}")
                        predicted_pockets[-1]['atoms'].append({
                            'atom_name': atom_name,
                            'coords': [x, y, z]
                        })
        
        # Calculate pocket centers
        for pocket in predicted_pockets:
            if pocket['atoms']:
                coords = np.array([atom['coords'] for atom in pocket['atoms']])
                pocket['center'] = coords.mean(axis=0).tolist()
                pocket['num_residues'] = len(set(pocket['residues']))
        
        return predicted_pockets
    
    def get_ground_truth_pocket(self, pdb_id):
        """Get ground truth pocket from ligand location"""
        ligand_file = self.data_dir / pdb_id / f"{pdb_id}_ligand.mol2"
        protein_file = self.data_dir / pdb_id / f"{pdb_id}_protein.pdb"
        
        if not ligand_file.exists() or not protein_file.exists():
            return []
        
        # Read ligand coordinates
        ligand_coords = []
        try:
            with open(ligand_file, 'r') as f:
                lines = f.readlines()
                in_atom_section = False
                for line in lines:
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
                                ligand_coords.append([x, y, z])
                            except:
                                pass
        except:
            # Try alternative format
            try:
                from rdkit import Chem
                mol = Chem.MolFromMol2File(str(ligand_file))
                if mol:
                    conf = mol.GetConformer()
                    for i in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(i)
                        ligand_coords.append([pos.x, pos.y, pos.z])
            except:
                pass
        
        if not ligand_coords:
            return []
        
        ligand_coords = np.array(ligand_coords)
        
        # Read protein residues
        from Bio import PDB
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(protein_file))
        
        ground_truth_residues = []
        
        for residue in structure.get_residues():
            # Get residue center (CA atom or average of atoms)
            atom_coords = []
            for atom in residue:
                atom_coords.append(atom.get_coord())
            
            if atom_coords:
                residue_center = np.mean(atom_coords, axis=0)
                
                # Check distance to any ligand atom
                distances = np.linalg.norm(ligand_coords - residue_center, axis=1)
                if np.min(distances) <= self.pocket_cutoff:
                    ground_truth_residues.append({
                        'chain': residue.parent.id if residue.parent else 'A',
                        'residue_num': residue.id[1],
                        'residue_name': residue.resname,
                        'center': residue_center.tolist()
                    })
        
        return ground_truth_residues
    
    def evaluate_pocket(self, predicted_pockets, ground_truth_residues):
        """Evaluate predicted pockets against ground truth"""
        if not predicted_pockets or not ground_truth_residues:
            return 0.0, 0.0, 0.0, [], []
        
        # Convert to sets for comparison
        gt_set = set([f"{r['chain']}:{r['residue_num']}:{r['residue_name']}" 
                     for r in ground_truth_residues])
        
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_pocket_idx = -1
        all_precisions = []
        all_recalls = []
        
        for i, pocket in enumerate(predicted_pockets):
            if pocket['num_residues'] < self.min_pocket_residues:
                continue
            
            pred_set = set(pocket['residues'])
            
            # Calculate metrics
            tp = len(pred_set.intersection(gt_set))
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            if tp + fp + fn > 0:
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    best_pocket_idx = i
        
        # If no pocket matched, use average of top predictions
        if best_pocket_idx == -1 and all_precisions:
            best_precision = np.mean(all_precisions[:3]) if len(all_precisions) >= 3 else np.mean(all_precisions)
            best_recall = np.mean(all_recalls[:3]) if len(all_recalls) >= 3 else np.mean(all_recalls)
            best_f1 = 2 * best_precision * best_recall / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0.0
        
        return best_f1, best_precision, best_recall, all_precisions, all_recalls
    
    def run_baseline(self, pdb_ids, split_name="train"):
        """Run baseline on a set of proteins"""
        print(f"\nRunning FPOCKET baseline on {split_name} set ({len(pdb_ids)} proteins)")
        
        results = []
        failed = []
        
        for idx, pdb_id in enumerate(pdb_ids):
            print(f"\n[{idx+1}/{len(pdb_ids)}] Processing {pdb_id}")
            
            # Run FPOCKET
            success, output = self.run_fpocket(pdb_id)
            
            if not success:
                failed.append(pdb_id)
                continue
            
            # Parse output
            predicted_pockets = self.parse_fpocket_output(pdb_id, output)
            
            # Get ground truth
            ground_truth = self.get_ground_truth_pocket(pdb_id)
            
            # Evaluate
            f1, precision, recall, all_prec, all_rec = self.evaluate_pocket(
                predicted_pockets, ground_truth
            )
            
            results.append({
                'pdb_id': pdb_id,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'num_predicted_pockets': len(predicted_pockets),
                'num_ground_truth_residues': len(ground_truth),
                'all_precisions': all_prec,
                'all_recalls': all_rec,
                'success': True
            })
            
            print(f"  Results: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        # Save results
        df = pd.DataFrame(results)
        output_file = self.results_dir / f"fpocket_{split_name}_results.csv"
        df.to_csv(output_file, index=False)
        
        # Save failed list
        if failed:
            failed_file = self.results_dir / f"fpocket_{split_name}_failed.txt"
            with open(failed_file, 'w') as f:
                f.write("\n".join(failed))
        
        return results, failed
    
    def compute_overall_metrics(self):
        """Compute overall performance metrics"""
        # Load all results
        train_file = self.results_dir / "fpocket_train_results.csv"
        val_file = self.results_dir / "fpocket_val_results.csv"
        test_file = self.results_dir / "fpocket_test_results.csv"
        
        all_results = []
        
        for file_path, split_name in [(train_file, 'train'), (val_file, 'val'), (test_file, 'test')]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['split'] = split_name
                all_results.append(df)
        
        if not all_results:
            print("No results found!")
            return None
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Compute overall metrics
        overall_metrics = {
            'overall_f1': combined_df['f1'].mean(),
            'overall_precision': combined_df['precision'].mean(),
            'overall_recall': combined_df['recall'].mean(),
            'train_f1': combined_df[combined_df['split'] == 'train']['f1'].mean() if 'train' in combined_df['split'].unique() else 0,
            'val_f1': combined_df[combined_df['split'] == 'val']['f1'].mean() if 'val' in combined_df['split'].unique() else 0,
            'test_f1': combined_df[combined_df['split'] == 'test']['f1'].mean() if 'test' in combined_df['split'].unique() else 0,
            'per_protein_f1': combined_df['f1'].tolist(),
            'per_protein_precision': combined_df['precision'].tolist(),
            'per_protein_recall': combined_df['recall'].tolist(),
            'num_proteins_evaluated': len(combined_df),
            'num_failed': len(self.train_ids) + len(self.val_ids) + len(self.test_ids) - len(combined_df)
        }
        
        # Save metrics
        metrics_file = self.results_dir / "fpocket_performance.json"
        with open(metrics_file, 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        
        # Create LaTeX table
        self.create_latex_table(overall_metrics)
        
        return overall_metrics
    
    def create_latex_table(self, metrics):
        """Create LaTeX table for paper"""
        latex = """\\begin{table}[h]
\\centering
\\caption{FPOCKET Baseline Performance on PDBbind Dataset}
\\label{tab:fpocket_baseline}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Split} & \\textbf{F1 Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\midrule
"""
        
        splits = ['train', 'val', 'test']
        for split in splits:
            f1_key = f'{split}_f1'
            if f1_key in metrics:
                latex += f"{split.capitalize()} & {metrics[f1_key]:.3f} & {metrics[f'{split}_precision']:.3f} & {metrics[f'{split}_recall']:.3f} \\\\\n"
        
        latex += f"""\\midrule
Overall & {metrics['overall_f1']:.3f} & {metrics['overall_precision']:.3f} & {metrics['overall_recall']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        
        latex_file = self.results_dir / "baseline_comparison.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        
        print(f"\nLaTeX table saved to: {latex_file}")
    
    def run_full_baseline(self):
        """Run full baseline evaluation"""
        print("=== FPOCKET Baseline Evaluation ===")
        print(f"Total proteins: {len(self.train_ids) + len(self.val_ids) + len(self.test_ids)}")
        print(f"Train: {len(self.train_ids)}, Val: {len(self.val_ids)}, Test: {len(self.test_ids)}")
        
        start_time = time.time()
        
        # Run on all splits
        self.run_baseline(self.train_ids[:10], "train")  # Start with 10 for testing
        # self.run_baseline(self.val_ids, "val")  # Uncomment for full run
        # self.run_baseline(self.test_ids, "test")  # Uncomment for full run
        
        # Compute overall metrics
        metrics = self.compute_overall_metrics()
        
        elapsed_time = time.time() - start_time
        
        if metrics:
            print(f"\n=== BASELINE RESULTS ===")
            print(f"Overall F1: {metrics['overall_f1']:.3f}")
            print(f"Overall Precision: {metrics['overall_precision']:.3f}")
            print(f"Overall Recall: {metrics['overall_recall']:.3f}")
            print(f"Proteins evaluated: {metrics['num_proteins_evaluated']}")
            print(f"Failed: {metrics['num_failed']}")
            print(f"Elapsed time: {elapsed_time:.1f} seconds")
            
            # Check against literature (F1 ~0.52)
            if 0.47 <= metrics['overall_f1'] <= 0.57:
                print("\n✅ SUCCESS: Baseline matches literature (F1 ≈ 0.52)")
                print("TARGET 2: ✅ fpocket_performance.json created")
            else:
                print(f"\n⚠️ WARNING: Baseline F1 ({metrics['overall_f1']:.3f}) outside expected range 0.47-0.57")
                print("Check ground truth definition and evaluation")
        
        return metrics

def main():
    baseline = FPocketBaseline()
    metrics = baseline.run_full_baseline()
    
    if metrics:
        print(f"\nResults saved to: ./experiments/results/phase1/")
        print("- fpocket_performance.json")
        print("- baseline_comparison.tex")
        print("- fpocket_*_results.csv")

if __name__ == "__main__":
    main()
