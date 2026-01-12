#!/usr/bin/env python3
"""
EMERGENCY: Process 200 proteins quickly for Phase 2C
"""
import os
import torch
from tqdm import tqdm
import sys
sys.path.append('./src/models')

# Try to import the working builder from Phase 2B
try:
    from simple_enhanced_builder import SimpleEnhancedGraphBuilder as Builder
    print("Using SimpleEnhancedGraphBuilder")
except:
    # Fallback to simplest builder
    print("Creating simplest builder...")
    import numpy as np
    from torch_geometric.data import Data
    
    class EmergencyBuilder:
        def __init__(self, cutoff=8.0):
            self.cutoff = cutoff
            self.residue_types = {
                'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
            }
        
        def build_from_pdb(self, pdb_file, ligand_file=None, protein_id=None):
            """Super simple PDB parser"""
            try:
                # Parse residues
                residues = []
                current_res = None
                coords = []
                
                with open(pdb_file, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM'):
                            res_name = line[17:20].strip()
                            res_id = int(line[22:26].strip())
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            
                            if current_res != res_id:
                                if current_res is not None and coords:
                                    centroid = np.mean(coords, axis=0)
                                    residues.append({
                                        'name': current_res_name,
                                        'id': current_res,
                                        'centroid': centroid,
                                        'coords': coords
                                    })
                                current_res = res_id
                                current_res_name = res_name
                                coords = []
                            
                            coords.append([x, y, z])
                
                # Add last residue
                if current_res is not None and coords:
                    centroid = np.mean(coords, axis=0)
                    residues.append({
                        'name': current_res_name,
                        'id': current_res,
                        'centroid': centroid,
                        'coords': coords
                    })
                
                if len(residues) == 0:
                    return None
                
                # Create features (30 dimensions - same as Phase 2B)
                features = []
                positions = []
                
                for res in residues:
                    # Residue type (20)
                    res_type = np.zeros(20)
                    if res['name'] in self.residue_types:
                        res_type[self.residue_types[res['name']]] = 1.0
                    
                    # Position (3)
                    centroid = res['centroid']
                    
                    # Simple properties (7)
                    props = self.get_simple_properties(res['name'])
                    
                    # Combine (30 features total)
                    feat = list(res_type) + [centroid[0], centroid[1], centroid[2]] + props
                    features.append(feat)
                    positions.append(centroid)
                
                features = torch.tensor(features, dtype=torch.float)
                positions = torch.tensor(positions, dtype=torch.float)
                
                # Create edges (simple distance-based)
                num_nodes = len(residues)
                edges = []
                
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        dist = torch.norm(positions[i] - positions[j])
                        if dist < self.cutoff:
                            edges.append([i, j])
                            edges.append([j, i])
                
                edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
                
                # Labels (if ligand provided)
                if ligand_file and os.path.exists(ligand_file):
                    labels = self.create_labels(residues, ligand_file)
                else:
                    labels = torch.zeros(num_nodes)
                
                # Create graph
                graph = Data(
                    x=features,
                    edge_index=edge_index,
                    y=labels,
                    pos=positions,
                    protein_id=protein_id or os.path.basename(pdb_file).split('_')[0]
                )
                
                return graph
                
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
                return None
        
        def get_simple_properties(self, res_name):
            """7 simple properties"""
            # hydrophobicity, charge, polarity, size, flexibility, accessibility, aromaticity
            props_dict = {
                'ALA': [0.62, 0, 0, 0.31, 0.4, 0.7, 0],
                'ARG': [-2.53, 1, 1, 1.01, 0.3, 0.5, 0],
                'ASN': [-0.78, 0, 1, 0.60, 0.5, 0.6, 0],
                'ASP': [-0.90, -1, 1, 0.60, 0.4, 0.6, 0],
                'CYS': [0.29, 0, 0, 0.55, 0.3, 0.4, 0],
                'GLN': [-0.85, 0, 1, 0.72, 0.5, 0.6, 0],
                'GLU': [-0.74, -1, 1, 0.72, 0.4, 0.6, 0],
                'GLY': [0.48, 0, 0, 0.00, 0.6, 0.8, 0],
                'HIS': [-0.40, 0.5, 1, 0.78, 0.4, 0.5, 1],
                'ILE': [1.38, 0, 0, 0.96, 0.3, 0.4, 0],
                'LEU': [1.06, 0, 0, 0.96, 0.3, 0.4, 0],
                'LYS': [-1.50, 1, 1, 0.84, 0.4, 0.5, 0],
                'MET': [0.64, 0, 0, 0.75, 0.4, 0.5, 0],
                'PHE': [1.19, 0, 0, 1.12, 0.3, 0.3, 1],
                'PRO': [0.12, 0, 0, 0.72, 0.2, 0.5, 0],
                'SER': [-0.18, 0, 1, 0.38, 0.6, 0.7, 0],
                'THR': [-0.05, 0, 1, 0.55, 0.5, 0.6, 0],
                'TRP': [0.81, 0, 0, 1.37, 0.3, 0.3, 1],
                'TYR': [0.26, 0, 1, 1.14, 0.4, 0.4, 1],
                'VAL': [1.08, 0, 0, 0.72, 0.3, 0.5, 0]
            }
            return props_dict.get(res_name, [0, 0, 0, 0.5, 0.5, 0.5, 0])
        
        def create_labels(self, residues, ligand_file):
            """Simple label creation"""
            # Parse ligand coords (simplified)
            lig_coords = []
            try:
                with open(ligand_file, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') or (len(line.split()) >= 4):
                            parts = line.split()
                            try:
                                if len(parts) >= 3:
                                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                                    lig_coords.append([x, y, z])
                            except:
                                continue
            except:
                pass
            
            if not lig_coords:
                return torch.zeros(len(residues))
            
            labels = []
            lig_coords = np.array(lig_coords)
            
            for res in residues:
                res_coords = np.array(res['coords'])
                # Minimum distance between residue and ligand
                distances = np.linalg.norm(res_coords[:, np.newaxis] - lig_coords, axis=2)
                min_dist = distances.min()
                labels.append(1.0 if min_dist < 4.0 else 0.0)
            
            return torch.tensor(labels, dtype=torch.float)
    
    Builder = EmergencyBuilder

def process_200_proteins():
    """Process 200 proteins quickly"""
    builder = Builder(cutoff_distance=8.0)
    
    # Read train IDs
    with open('./experiments/results/phase1/splits/train_ids.txt', 'r') as f:
        all_ids = [line.strip() for line in f if line.strip()]
    
    # Take first 200
    target_ids = all_ids[:200]
    print(f"Processing {len(target_ids)} proteins...")
    
    processed = []
    output_dir = './data/processed/emergency_200'
    os.makedirs(output_dir, exist_ok=True)
    
    for pid in tqdm(target_ids, desc="Processing"):
        protein_path = f"./data/PDBbind/refined-set/{pid}/{pid}_protein.pdb"
        ligand_path = f"./data/PDBbind/refined-set/{pid}/{pid}_ligand.mol2"
        
        if not os.path.exists(ligand_path):
            ligand_path = f"./data/PDBbind/refined-set/{pid}/{pid}_ligand.sdf"
        
        if os.path.exists(protein_path):
            graph = builder.build_from_pdb(protein_path, ligand_path, pid)
            if graph is not None:
                processed.append(graph)
                torch.save(graph, f"{output_dir}/{pid}_graph.pt")
    
    print(f"\nSuccessfully processed {len(processed)}/{len(target_ids)} proteins")
    print(f"Graphs saved to: {output_dir}/")
    
    # Save statistics
    stats = {
        'total_processed': len(processed),
        'target_count': len(target_ids),
        'success_rate': len(processed)/len(target_ids),
        'feature_dim': processed[0].x.shape[1] if processed else 0
    }
    
    import json
    with open(f'{output_dir}/processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return processed

if __name__ == "__main__":
    process_200_proteins()
