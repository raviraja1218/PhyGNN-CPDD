"""
SIMPLE BUILDER - FIXED VERSION
For fixing labels in Phase 2C
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data

class SimpleBuilderFixed:
    def __init__(self, cutoff=8.0):
        self.cutoff = cutoff
        
        # Residue types
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
    
    def build_from_pdb(self, pdb_file, ligand_file=None, protein_id=None):
        """Build graph from PDB with CORRECT labels"""
        try:
            # Parse protein
            residues = self.parse_pdb(pdb_file)
            if not residues:
                return None
            
            # Create features (30D like Phase 2B)
            features, positions = self.create_features(residues)
            
            # Create edges
            edge_index = self.create_edges(positions)
            
            # Create CORRECT labels from ligand
            labels = self.create_correct_labels(residues, ligand_file)
            
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
            print(f"Error in SimpleBuilderFixed: {e}")
            return None
    
    def parse_pdb(self, pdb_file):
        """Parse PDB file"""
        residues = []
        current_res = None
        coords = []
        current_res_name = ""
        
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
                                'coords': np.array(coords)
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
                'coords': np.array(coords)
            })
        
        return residues
    
    def create_features(self, residues):
        """Create 30D features (matching Phase 2B)"""
        features = []
        positions = []
        
        for res in residues:
            # Residue type (20D)
            res_type = np.zeros(20)
            if res['name'] in self.residue_types:
                res_type[self.residue_types[res['name']]] = 1.0
            
            # Position (3D)
            centroid = res['centroid']
            
            # Properties (7D)
            props = self.get_properties(res['name'])
            
            # Combine (30D)
            feat = np.concatenate([res_type, centroid, props])
            features.append(feat)
            positions.append(centroid)
        
        features = torch.tensor(np.array(features), dtype=torch.float)
        positions = torch.tensor(np.array(positions), dtype=torch.float)
        
        return features, positions
    
    def get_properties(self, res_name):
        """Get 7 properties"""
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
    
    def create_edges(self, positions):
        """Create edges based on distance"""
        num_nodes = positions.shape[0]
        edges = []
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = torch.norm(positions[i] - positions[j])
                if dist < self.cutoff:
                    edges.append([i, j])
                    edges.append([j, i])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    def create_correct_labels(self, residues, ligand_file):
        """Create CORRECT pocket labels from ligand"""
        if not ligand_file or not os.path.exists(ligand_file):
            return torch.zeros(len(residues))
        
        # Parse ligand coordinates
        lig_coords = self.parse_ligand_coords(ligand_file)
        if len(lig_coords) == 0:
            return torch.zeros(len(residues))
        
        lig_coords = np.array(lig_coords)
        
        # Label residues within 4Å of ANY ligand atom
        labels = []
        for res in residues:
            if len(res['coords']) == 0:
                labels.append(0.0)
                continue
            
            # Calculate minimum distance
            distances = np.sqrt(((res['coords'][:, np.newaxis, :] - lig_coords) ** 2).sum(axis=2))
            min_dist = distances.min()
            labels.append(1.0 if min_dist < 4.0 else 0.0)
        
        return torch.tensor(labels, dtype=torch.float)
    
    def parse_ligand_coords(self, ligand_file):
        """Parse ligand coordinates from various formats"""
        coords = []
        
        try:
            with open(ligand_file, 'r') as f:
                lines = f.readlines()
            
            # Try different formats
            for line in lines:
                # PDB format
                if line.startswith('HETATM') or line.startswith('ATOM'):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except:
                        pass
                # SDF/MOL2 format (space separated)
                elif len(line.split()) >= 4:
                    parts = line.split()
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coords.append([x, y, z])
                    except:
                        pass
        
        except:
            pass
        
        return coords

# Test
if __name__ == "__main__":
    builder = SimpleBuilderFixed(cutoff=8.0)
    print("SimpleBuilderFixed created successfully")
