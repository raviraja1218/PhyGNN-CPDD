"""
Fixed Graph Builder with Proper Pocket Labeling
"""
import torch
import numpy as np
import os
from torch_geometric.data import Data

class GraphBuilderFixed:
    def __init__(self, cutoff_distance=8.0, pocket_cutoff=4.0):
        self.cutoff = cutoff_distance
        self.pocket_cutoff = pocket_cutoff
        
        # Residue type mapping
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
    
    def parse_pdb(self, pdb_file):
        """Parse PDB file and extract residue information"""
        residues = []
        current_residue = None
        residue_atoms = []
        prev_res_name = ""
        prev_chain = ""
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # Extract basic info
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
                                centroid = np.mean(residue_atoms, axis=0)
                                residues.append({
                                    'name': prev_res_name,
                                    'id': current_residue,
                                    'chain': prev_chain,
                                    'centroid': centroid,
                                    'atoms': np.array(residue_atoms)
                                })
                            
                            # Start new residue
                            current_residue = res_id
                            prev_res_name = res_name
                            prev_chain = chain_id
                            residue_atoms = []
                        
                        residue_atoms.append([x, y, z])
            
            # Save last residue
            if current_residue is not None and residue_atoms:
                centroid = np.mean(residue_atoms, axis=0)
                residues.append({
                    'name': prev_res_name,
                    'id': current_residue,
                    'chain': prev_chain,
                    'centroid': centroid,
                    'atoms': np.array(residue_atoms)
                })
                
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
        
        return residues
    
    def parse_ligand_coords(self, ligand_file):
        """Parse ligand file to get coordinates"""
        coords = []
        
        try:
            with open(ligand_file, 'r') as f:
                lines = f.readlines()
            
            # Try different formats
            if ligand_file.endswith('.mol2'):
                in_atom_section = False
                for line in lines:
                    if '@<TRIPOS>ATOM' in line:
                        in_atom_section = True
                        continue
                    elif '@<TRIPOS>' in line and in_atom_section:
                        break
                    
                    if in_atom_section and line.strip():
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                                coords.append([x, y, z])
                            except:
                                continue
            
            elif ligand_file.endswith('.sdf'):
                # Parse SDF format
                for line in lines[4:]:  # Skip header
                    if line.strip() and len(line.split()) >= 4:
                        parts = line.split()
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            coords.append([x, y, z])
                        except:
                            continue
            
            elif ligand_file.endswith('.pdb'):
                # Parse PDB format
                for line in lines:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords.append([x, y, z])
                        except:
                            continue
        
        except Exception as e:
            print(f"Warning parsing ligand {ligand_file}: {e}")
        
        return np.array(coords) if coords else np.array([])
    
    def create_pocket_labels(self, residues, ligand_coords):
        """Create binary labels for pocket residues"""
        if len(ligand_coords) == 0:
            return np.zeros(len(residues))
        
        labels = []
        
        for res in residues:
            # Get all atom coordinates for this residue
            res_coords = res['atoms']  # Shape: [num_atoms, 3]
            
            # Calculate minimum distance between any residue atom and any ligand atom
            min_dist = float('inf')
            for res_atom in res_coords:
                for lig_atom in ligand_coords:
                    dist = np.linalg.norm(res_atom - lig_atom)
                    if dist < min_dist:
                        min_dist = dist
            
            # Label as pocket if any atom is within pocket_cutoff of ligand
            labels.append(1.0 if min_dist < self.pocket_cutoff else 0.0)
        
        return np.array(labels)
    
    def create_graph(self, pdb_file, ligand_file, protein_id="test"):
        """Create PyTorch Geometric graph with proper pocket labels"""
        # Parse residues
        residues = self.parse_pdb(pdb_file)
        
        if not residues:
            print(f"No residues found in {pdb_file}")
            return None
        
        # Parse ligand coordinates
        ligand_coords = self.parse_ligand_coords(ligand_file)
        
        if len(ligand_coords) == 0:
            print(f"No ligand coordinates found in {ligand_file}")
            return None
        
        print(f"  Ligand atoms: {len(ligand_coords)}")
        
        # Create node features
        node_features = []
        positions = []
        
        for res in residues:
            # One-hot encoding for residue type (20 dimensions)
            one_hot = np.zeros(20)
            if res['name'] in self.residue_types:
                one_hot[self.residue_types[res['name']]] = 1.0
            
            # Add centroid coordinates (3 dimensions)
            # Add number of atoms (1 dimension, normalized)
            # Add chain as one-hot (simplified)
            chain_feat = 1.0 if res['chain'] != 'A' else 0.0
            
            features = list(one_hot) + list(res['centroid']) + [len(res['atoms'])/20.0, chain_feat]
            node_features.append(features)
            positions.append(res['centroid'])
        
        # Convert to tensors (fixing the warning)
        node_features_array = np.array(node_features, dtype=np.float32)
        x = torch.from_numpy(node_features_array)
        
        positions_array = np.array(positions, dtype=np.float32)
        pos = torch.from_numpy(positions_array)
        
        # Create edges (distance-based)
        edges = []
        num_residues = len(residues)
        
        # Use numpy for faster distance calculation
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                dist = np.linalg.norm(positions_array[i] - positions_array[j])
                if dist < self.cutoff:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create PROPER pocket labels
        y_np = self.create_pocket_labels(residues, ligand_coords)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        # Count pocket residues
        pocket_count = int(y.sum().item())
        print(f"  Pocket residues: {pocket_count}/{len(residues)} ({pocket_count/len(residues)*100:.1f}%)")
        
        # Create PyG Data object
        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos=pos,
            protein_id=protein_id,
            num_nodes=len(residues)
        )
        
        return graph
    
    def save_graph(self, graph, output_path):
        """Save graph to file"""
        torch.save(graph, output_path)
    
    def load_graph(self, input_path):
        """Load graph from file"""
        return torch.load(input_path, weights_only=False)

def test_fixed_builder():
    """Test the fixed graph builder"""
    builder = GraphBuilderFixed(cutoff_distance=8.0, pocket_cutoff=4.0)
    
    # Find a sample protein
    sample_dir = "./data/PDBbind/refined-set"
    
    # Look for first available protein with ligand
    import os
    for item in os.listdir(sample_dir):
        protein_path = os.path.join(sample_dir, item, f"{item}_protein.pdb")
        ligand_path = os.path.join(sample_dir, item, f"{item}_ligand.mol2")
        
        if not os.path.exists(ligand_path):
            ligand_path = os.path.join(sample_dir, item, f"{item}_ligand.sdf")
        
        if os.path.exists(protein_path) and os.path.exists(ligand_path):
            print(f"\nTesting on {item}")
            print(f"  Protein: {protein_path}")
            print(f"  Ligand: {ligand_path}")
            
            # Create graph
            graph = builder.create_graph(protein_path, ligand_path, item)
            
            if graph is not None:
                print(f"\n✓ Graph created successfully!")
                print(f"  Nodes: {graph.num_nodes}")
                print(f"  Edges: {graph.edge_index.shape[1]}")
                print(f"  Features: {graph.x.shape[1]}")
                print(f"  Pocket labels: {graph.y.sum().item()}/{graph.num_nodes}")
                
                # Save graph
                output_dir = "./data/processed/graphs/samples_fixed"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{item}_graph_fixed.pt")
                builder.save_graph(graph, output_path)
                print(f"  Saved to: {output_path}")
                
                # Show some statistics
                y_np = graph.y.numpy()
                print(f"  Class distribution: {np.sum(y_np==0)} non-pocket, {np.sum(y_np==1)} pocket")
                
                return True
    
    print("✗ No suitable protein-ligand pair found for testing")
    return False

if __name__ == "__main__":
    print("Testing Fixed Graph Builder with Proper Pocket Labels...")
    success = test_fixed_builder()
    if success:
        print("\n✓ Fixed Graph Builder test PASSED!")
    else:
        print("\n✗ Fixed Graph Builder test FAILED")
