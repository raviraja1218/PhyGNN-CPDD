"""
Simple Enhanced Graph Builder (without radius_gyration error)
"""
import torch
import numpy as np
import os
from torch_geometric.data import Data

class SimpleEnhancedBuilder:
    def __init__(self, cutoff_distance=8.0, pocket_cutoff=4.0):
        self.cutoff = cutoff_distance
        self.pocket_cutoff = pocket_cutoff
        
        # Basic residue properties
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
            'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
            'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }
        
        self.charge = {
            'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
            'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.5, 'ILE': 0,
            'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
            'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
        }
    
    def parse_pdb_simple(self, pdb_file):
        """Simple PDB parser without radius_gyration"""
        residues = []
        current_residue = None
        residue_atoms = []
        prev_res_name = ""
        prev_chain = ""
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # Extract atom info
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        res_id = int(line[22:26].strip())
                        chain_id = line[21]
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        if current_residue != res_id or chain_id != prev_chain:
                            # Save previous residue
                            if current_residue is not None and residue_atoms:
                                coords_array = np.array(residue_atoms)
                                centroid = coords_array.mean(axis=0)
                                
                                residues.append({
                                    'name': prev_res_name,
                                    'id': current_residue,
                                    'chain': prev_chain,
                                    'centroid': centroid,
                                    'atoms': coords_array,
                                    'num_atoms': len(residue_atoms)
                                })
                            
                            # Start new residue
                            current_residue = res_id
                            prev_res_name = res_name
                            prev_chain = chain_id
                            residue_atoms = []
                        
                        residue_atoms.append([x, y, z])
            
            # Save last residue
            if current_residue is not None and residue_atoms:
                coords_array = np.array(residue_atoms)
                centroid = coords_array.mean(axis=0)
                
                residues.append({
                    'name': prev_res_name,
                    'id': current_residue,
                    'chain': prev_chain,
                    'centroid': centroid,
                    'atoms': coords_array,
                    'num_atoms': len(residue_atoms)
                })
                
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
        
        return residues
    
    def parse_ligand_coords(self, ligand_file):
        """Parse ligand coordinates"""
        coords = []
        
        try:
            with open(ligand_file, 'r') as f:
                lines = f.readlines()
            
            # Simple parsing
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coords.append([x, y, z])
                    except:
                        continue
        
        except Exception as e:
            print(f"Warning parsing ligand {ligand_file}: {e}")
        
        return np.array(coords) if coords else np.array([])
    
    def create_pocket_labels(self, residues, ligand_coords):
        """Create pocket labels"""
        if len(ligand_coords) == 0:
            return np.zeros(len(residues))
        
        labels = []
        
        for res in residues:
            # Calculate min distance between residue atoms and ligand atoms
            res_coords = res['atoms']
            min_dist = float('inf')
            
            # Vectorized distance calculation
            for res_atom in res_coords:
                dists = np.linalg.norm(ligand_coords - res_atom, axis=1)
                min_dist = min(min_dist, dists.min())
            
            labels.append(1.0 if min_dist < self.pocket_cutoff else 0.0)
        
        return np.array(labels)
    
    def create_enhanced_graph(self, pdb_file, ligand_file, protein_id="test"):
        """Create graph with enhanced but simple features"""
        # Parse residues
        residues = self.parse_pdb_simple(pdb_file)
        
        if not residues:
            print(f"No residues found in {pdb_file}")
            return None
        
        # Parse ligand
        ligand_coords = self.parse_ligand_coords(ligand_file)
        
        if len(ligand_coords) == 0:
            print(f"No ligand coordinates found in {ligand_file}")
            # Try alternative: use any ligand file in directory
            ligand_dir = os.path.dirname(ligand_file)
            for fname in os.listdir(ligand_dir):
                if 'ligand' in fname.lower():
                    alt_path = os.path.join(ligand_dir, fname)
                    if alt_path != ligand_file:
                        ligand_coords = self.parse_ligand_coords(alt_path)
                        if len(ligand_coords) > 0:
                            print(f"  Using alternative ligand file: {fname}")
                            break
        
        if len(ligand_coords) == 0:
            print(f"  WARNING: No ligand found for {protein_id}")
            return None
        
        print(f"  {protein_id}: {len(residues)} residues, {len(ligand_coords)} ligand atoms")
        
        # Create enhanced features (simplified)
        node_features = []
        positions = []
        
        for i, res in enumerate(residues):
            features = []
            
            # 1. Residue type one-hot (20 dim)
            aa_list = list(self.hydrophobicity.keys())
            one_hot = np.zeros(20)
            if res['name'] in aa_list:
                one_hot[aa_list.index(res['name'])] = 1.0
            features.extend(one_hot)
            
            # 2. Physicochemical properties (2 dim)
            hydrophobicity = self.hydrophobicity.get(res['name'], 0.0)
            charge = self.charge.get(res['name'], 0.0)
            features.extend([hydrophobicity, charge])
            
            # 3. Structural features (4 dim)
            centroid = res['centroid']
            features.extend([centroid[0], centroid[1], centroid[2]])  # Position
            features.append(res['num_atoms'] / 20.0)  # Normalized atom count
            
            # 4. Positional encoding (5 dim)
            norm_pos = i / len(residues)
            features.append(norm_pos)
            features.append(np.sin(norm_pos * 2 * np.pi))
            features.append(np.cos(norm_pos * 2 * np.pi))
            
            # 5. Chain encoding (1 dim - simplified)
            chain_code = 1.0 if res['chain'] != 'A' else 0.0
            features.append(chain_code)
            
            node_features.append(features)
            positions.append(centroid)
        
        # Convert to tensors
        node_features_array = np.array(node_features, dtype=np.float32)
        x = torch.from_numpy(node_features_array)
        
        positions_array = np.array(positions, dtype=np.float32)
        pos = torch.from_numpy(positions_array)
        
        # Create edges (distance-based, optimized)
        n = len(positions)
        edges = []
        
        # Simple edge creation (could be optimized)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions_array[i] - positions_array[j])
                if dist < self.cutoff:
                    edges.append([i, j])
                    edges.append([j, i])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create pocket labels
        y_np = self.create_pocket_labels(residues, ligand_coords)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        pocket_count = int(y.sum().item())
        pocket_percent = pocket_count / n * 100 if n > 0 else 0
        
        if pocket_count == 0:
            print(f"  WARNING: No pockets found for {protein_id}")
            return None
        
        print(f"  Pocket residues: {pocket_count}/{n} ({pocket_percent:.1f}%)")
        
        # Create graph
        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos=pos,
            protein_id=protein_id,
            num_nodes=n,
            num_features=x.shape[1]
        )
        
        return graph
    
    def save_graph(self, graph, output_path):
        """Save graph"""
        torch.save(graph, output_path)

def test_simple_builder():
    """Test the simple enhanced builder"""
    print("Testing Simple Enhanced Builder...")
    
    builder = SimpleEnhancedBuilder()
    
    # Find a sample protein
    sample_dir = "./data/PDBbind/refined-set"
    
    for item in os.listdir(sample_dir)[:5]:
        protein_path = os.path.join(sample_dir, item, f"{item}_protein.pdb")
        ligand_path = os.path.join(sample_dir, item, f"{item}_ligand.mol2")
        
        if not os.path.exists(ligand_path):
            ligand_path = os.path.join(sample_dir, item, f"{item}_ligand.sdf")
        
        if os.path.exists(protein_path) and os.path.exists(ligand_path):
            print(f"\nProcessing {item}...")
            graph = builder.create_enhanced_graph(protein_path, ligand_path, item)
            
            if graph is not None:
                print(f"✓ Graph created: {graph.num_nodes} nodes, {graph.num_features} features")
                print(f"  Pocket residues: {graph.y.sum().item()}/{graph.num_nodes}")
                
                # Save sample
                output_dir = "./data/processed/graphs_simple_enhanced"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{item}_graph.pt")
                builder.save_graph(graph, output_path)
                print(f"  Saved to: {output_path}")
                return True
    
    print("✗ No suitable protein found")
    return False

if __name__ == "__main__":
    test_simple_builder()
