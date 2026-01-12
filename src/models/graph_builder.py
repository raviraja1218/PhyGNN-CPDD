"""
Minimal Graph Builder for PhyGNN-CPDD
Creates protein graphs from PDB files
"""
import torch
import numpy as np
import os
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, cutoff_distance=8.0):
        self.cutoff = cutoff_distance
        
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
    
    def create_graph(self, pdb_file, ligand_file=None, protein_id="test"):
        """Create PyTorch Geometric graph from PDB file"""
        # Parse residues
        residues = self.parse_pdb(pdb_file)
        
        if not residues:
            print(f"No residues found in {pdb_file}")
            return None
        
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
            # Add chain as one-hot (simplified: 0 for A, 1 for others)
            chain_feat = 1.0 if res['chain'] != 'A' else 0.0
            
            features = list(one_hot) + list(res['centroid']) + [len(res['atoms'])/20.0, chain_feat]
            node_features.append(features)
            positions.append(res['centroid'])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        # Create edges (distance-based)
        edges = []
        num_residues = len(residues)
        
        # Use numpy for faster distance calculation
        pos_np = np.array(positions)
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                dist = np.linalg.norm(pos_np[i] - pos_np[j])
                if dist < self.cutoff:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create labels (all zeros for now - will add ligand parsing later)
        y = torch.zeros(len(residues), dtype=torch.float32)
        
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

def test_builder():
    """Test the graph builder"""
    builder = GraphBuilder(cutoff_distance=8.0)
    
    # Find a sample protein
    sample_dir = "./data/PDBbind/refined-set"
    
    # Look for first available protein
    import os
    for item in os.listdir(sample_dir):
        protein_path = os.path.join(sample_dir, item, f"{item}_protein.pdb")
        if os.path.exists(protein_path):
            print(f"Testing on {item}")
            
            # Create graph
            graph = builder.create_graph(protein_path, protein_id=item)
            
            if graph is not None:
                print(f"✓ Graph created successfully!")
                print(f"  Nodes: {graph.num_nodes}")
                print(f"  Edges: {graph.edge_index.shape[1]}")
                print(f"  Features: {graph.x.shape[1]}")
                print(f"  Protein ID: {graph.protein_id}")
                
                # Save graph
                output_dir = "./data/processed/graphs/samples"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{item}_graph.pt")
                builder.save_graph(graph, output_path)
                print(f"  Saved to: {output_path}")
                return True
    
    print("✗ No protein files found for testing")
    return False

if __name__ == "__main__":
    print("Testing Graph Builder...")
    success = test_builder()
    if success:
        print("\n✓ Graph Builder test PASSED!")
    else:
        print("\n✗ Graph Builder test FAILED")
