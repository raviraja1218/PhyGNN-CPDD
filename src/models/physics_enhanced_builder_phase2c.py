"""
Physics-Enhanced Graph Builder for PDBbind proteins
Extends simple_enhanced_builder.py with physics features
"""
import torch
import numpy as np
from torch_geometric.data import Data
import os
from tqdm import tqdm
import math

class PhysicsEnhancedGraphBuilder:
    def __init__(self, cutoff_distance=8.0):
        """
        Initialize physics-enhanced graph builder
        
        Args:
            cutoff_distance: Spatial cutoff for edges (Å)
        """
        self.cutoff = cutoff_distance
        
        # Residue type mapping (20 standard amino acids)
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
        
        # Physics parameters for residues
        self.residue_physics = self._init_residue_physics()
    
    def _init_residue_physics(self):
        """Initialize physics parameters for each residue type"""
        # Format: [partial_charge, vdw_radius, hbond_donor, hbond_acceptor, hydrophobicity]
        physics = {
            'ALA': [0.00, 1.87, 0, 0, 1.8],     # Alanine
            'ARG': [1.00, 2.03, 4, 6, -4.5],     # Arginine (positive)
            'ASN': [0.00, 1.93, 2, 4, -3.5],     # Asparagine
            'ASP': [-1.00, 1.91, 1, 4, -3.5],    # Aspartic acid (negative)
            'CYS': [0.00, 1.91, 1, 1, 2.5],      # Cysteine
            'GLN': [0.00, 1.96, 2, 4, -3.5],     # Glutamine
            'GLU': [-1.00, 1.94, 1, 4, -3.5],    # Glutamic acid (negative)
            'GLY': [0.00, 1.73, 1, 2, -0.4],     # Glycine
            'HIS': [0.50, 2.00, 2, 4, -3.2],     # Histidine
            'ILE': [0.00, 2.04, 1, 1, 4.5],      # Isoleucine
            'LEU': [0.00, 2.04, 1, 1, 3.8],      # Leucine
            'LYS': [1.00, 2.06, 3, 2, -3.9],     # Lysine (positive)
            'MET': [0.00, 2.03, 1, 2, 1.9],      # Methionine
            'PHE': [0.00, 2.10, 1, 1, 2.8],      # Phenylalanine
            'PRO': [0.00, 1.95, 0, 1, -1.6],     # Proline
            'SER': [0.00, 1.86, 1, 2, -0.8],     # Serine
            'THR': [0.00, 1.90, 1, 2, -0.7],     # Threonine
            'TRP': [0.00, 2.15, 1, 2, -0.9],     # Tryptophan
            'TYR': [0.00, 2.07, 1, 3, -1.3],     # Tyrosine
            'VAL': [0.00, 1.99, 1, 1, 4.2]       # Valine
        }
        return physics
    
    def build_from_pdb(self, pdb_file, ligand_file=None, protein_id=None):
        """
        Build physics-enhanced graph from PDB file
        
        Returns:
            PyTorch Geometric Data object with physics features
        """
        # Parse PDB file and get atoms (more detailed than residue-level)
        atoms = self.parse_atoms_from_pdb(pdb_file)
        
        if len(atoms) == 0:
            print(f"Warning: No atoms found in {pdb_file}")
            return None
        
        # Group atoms by residue for residue-level features
        residues = self.group_atoms_by_residue(atoms)
        
        # Create node features (residue-level with physics)
        node_features = self.create_physics_node_features(residues)
        
        # Create edges with physics attributes
        edge_index, edge_attr, edge_physics = self.create_physics_edges(residues)
        
        # Create labels (pocket residues)
        if ligand_file and os.path.exists(ligand_file):
            labels = self.create_pocket_labels(residues, ligand_file)
        else:
            labels = torch.zeros(len(residues))
        
        # Get residue coordinates (centroid of each residue)
        positions = torch.tensor(np.array([r['centroid'] for r in residues]), dtype=torch.float)
        
        # Create graph with physics metadata
        graph = Data(
            x=node_features,                # [num_residues, node_feat_dim]
            edge_index=edge_index,          # [2, num_edges]
            edge_attr=edge_attr,            # [num_edges, edge_feat_dim]
            y=labels,                       # [num_residues]
            pos=positions,                  # [num_residues, 3]
            protein_id=protein_id or os.path.basename(pdb_file).split('_')[0],
            num_nodes=len(residues),
            # Physics-specific attributes
            edge_physics=edge_physics,      # [num_edges, physics_feat_dim]
            atom_positions=torch.tensor([atom['coords'] for atom in atoms], dtype=torch.float),  # [num_atoms, 3]
            residue_mapping=torch.tensor([atom['res_index'] for atom in atoms], dtype=torch.long)  # [num_atoms]
        )
        
        return graph
    
    def parse_atoms_from_pdb(self, pdb_file):
        """Parse PDB file to extract atom-level information"""
        atoms = []
        
        try:
            with open(pdb_file, 'r') as f:
                residue_counter = -1
                last_res_id = None
                
                for line in f:
                    if line.startswith('ATOM'):
                        # Extract atom information
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        res_id = int(line[22:26].strip())
                        chain_id = line[21]
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        # Get element from atom name
                        element = self.get_element_from_atom(atom_name, res_name)
                        
                        # Track residue changes
                        if last_res_id != res_id:
                            residue_counter += 1
                            last_res_id = res_id
                        
                        atoms.append({
                            'name': atom_name,
                            'element': element,
                            'res_name': res_name,
                            'res_id': res_id,
                            'res_index': residue_counter,
                            'chain': chain_id,
                            'coords': [x, y, z]
                        })
                        
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            return []
        
        return atoms
    
    def get_element_from_atom(self, atom_name, res_name):
        """Determine element from atom name"""
        # Common elements in proteins
        if atom_name.startswith('N'):
            return 'N'
        elif atom_name.startswith('CA'):
            return 'C'
        elif atom_name.startswith('C'):
            return 'C'
        elif atom_name.startswith('O'):
            return 'O'
        elif atom_name.startswith('S'):
            return 'S'
        else:
            # Default based on atom name
            if atom_name[0] in ['N', 'C', 'O', 'S', 'H', 'P']:
                return atom_name[0]
            else:
                return 'C'  # Default to carbon
    
    def group_atoms_by_residue(self, atoms):
        """Group atoms by residue for residue-level features"""
        residues = {}
        
        for atom in atoms:
            res_key = f"{atom['chain']}_{atom['res_id']}"
            
            if res_key not in residues:
                residues[res_key] = {
                    'name': atom['res_name'],
                    'id': atom['res_id'],
                    'chain': atom['chain'],
                    'atoms': [],
                    'coords': [],
                    'elements': [],
                    'atom_names': []
                }
            
            residues[res_key]['atoms'].append(atom)
            residues[res_key]['coords'].append(atom['coords'])
            residues[res_key]['elements'].append(atom['element'])
            residues[res_key]['atom_names'].append(atom['name'])
        
        # Calculate centroids and other residue properties
        residue_list = []
        for res_key, res_data in residues.items():
            coords_array = np.array(res_data['coords'])
            res_data['centroid'] = coords_array.mean(axis=0)
            res_data['atoms_array'] = coords_array
            residue_list.append(res_data)
        
        return residue_list
    
    def create_physics_node_features(self, residues):
        """
        Create physics-enhanced node features
        
        Returns: [num_residues, 35 features]
        """
        features = []
        
        for res in residues:
            feat = []
            
            # 1. Residue type (one-hot encoding, 20 dimensions)
            res_type = np.zeros(20)
            if res['name'] in self.residue_types:
                res_type[self.residue_types[res['name']]] = 1.0
            feat.extend(res_type)
            
            # 2. Physics properties (5 dimensions)
            phys_props = self.residue_physics.get(res['name'], [0, 1.8, 0, 0, 0])
            feat.extend(phys_props)
            
            # 3. Position (normalized coordinates, 3 dimensions)
            centroid = res['centroid']
            feat.extend([centroid[0], centroid[1], centroid[2]])
            
            # 4. Structural properties (4 dimensions)
            # Radius of gyration for the residue
            coords = np.array(res['coords'])
            centroid = coords.mean(axis=0)
            sq_dists = np.sum((coords - centroid) ** 2, axis=1)
            radius_gyration = np.sqrt(np.mean(sq_dists))
            feat.append(radius_gyration)
            
            # Number of atoms
            feat.append(len(res['atoms']) / 20.0)
            
            # Element composition (simplified)
            elements = res['elements']
            carbon_ratio = elements.count('C') / len(elements) if elements else 0
            nitrogen_ratio = elements.count('N') / len(elements) if elements else 0
            feat.extend([carbon_ratio, nitrogen_ratio])
            
            # 5. Additional properties (3 dimensions)
            # Hydrophobicity (from Kyte-Doolittle scale)
            hydrophobicity = self.get_hydrophobicity(res['name'])
            feat.append(hydrophobicity)
            
            # Charge (estimated)
            charge = self.estimate_charge(res['name'])
            feat.append(charge)
            
            # Volume (estimated from number of atoms)
            volume = len(res['atoms']) * 10.0  # Rough estimate in Å³
            feat.append(volume / 100.0)  # Normalized
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float)
    
    def get_hydrophobicity(self, res_name):
        """Get hydrophobicity from Kyte-Doolittle scale"""
        hydrophobicity = {
            'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
            'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
            'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
            'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
        }
        return hydrophobicity.get(res_name, 0.0)
    
    def estimate_charge(self, res_name):
        """Estimate charge at physiological pH"""
        # Positive: ARG, LYS, HIS
        # Negative: ASP, GLU
        if res_name in ['ARG', 'LYS']:
            return 1.0
        elif res_name == 'HIS':
            return 0.5  # Partially charged
        elif res_name in ['ASP', 'GLU']:
            return -1.0
        else:
            return 0.0
    
    def create_physics_edges(self, residues):
        """
        Create edges with physics attributes
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (inverse distance)
            edge_physics: [num_edges, 4] (physics features)
        """
        if len(residues) == 0:
            empty = torch.empty((2, 0), dtype=torch.long)
            return empty, empty, empty
        
        # Get centroids
        centroids = np.array([r['centroid'] for r in residues])
        
        edges = []
        edge_attrs = []
        edge_physics_list = []
        
        # Create distance matrix
        n_res = len(residues)
        for i in range(n_res):
            for j in range(i + 1, n_res):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                
                if dist < self.cutoff:
                    # Add both directions for undirected graph
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Edge attribute: inverse distance
                    edge_attr = [1.0 / (dist + 1e-6)]
                    edge_attrs.append(edge_attr)
                    edge_attrs.append(edge_attr)
                    
                    # Physics features for this edge
                    phys_feat = self.calculate_edge_physics(residues[i], residues[j], dist)
                    edge_physics_list.append(phys_feat)
                    edge_physics_list.append(phys_feat)  # Same for both directions
        
        if not edges:
            empty = torch.empty((2, 0), dtype=torch.long)
            return empty, empty, empty
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        edge_physics = torch.tensor(edge_physics_list, dtype=torch.float)
        
        return edge_index, edge_attr, edge_physics
    
    def calculate_edge_physics(self, res1, res2, distance):
        """
        Calculate physics features for an edge between two residues
        
        Returns: [4] features
        """
        features = []
        
        # 1. Electrostatic potential (Coulomb's law)
        q1 = self.estimate_charge(res1['name'])
        q2 = self.estimate_charge(res2['name'])
        electrostatic = (q1 * q2) / (distance + 1e-6)  # Simplified
        features.append(electrostatic)
        
        # 2. van der Waals interaction (Lennard-Jones attractive term)
        r1 = self.residue_physics.get(res1['name'], [0, 1.8])[1]  # vdW radius
        r2 = self.residue_physics.get(res2['name'], [0, 1.8])[1]
        vdw_radius_sum = r1 + r2
        # Simplified: attractive when close, repulsive when overlapping
        vdw_interaction = -1.0 * (vdw_radius_sum / (distance + 1e-6)) ** 6
        features.append(vdw_interaction)
        
        # 3. Hydrogen bond potential
        hbond1 = self.residue_physics.get(res1['name'], [0, 0, 0, 0])[2:4]  # donor, acceptor
        hbond2 = self.residue_physics.get(res2['name'], [0, 0, 0, 0])[2:4]
        hbond_potential = 0
        if (hbond1[0] > 0 and hbond2[1] > 0) or (hbond1[1] > 0 and hbond2[0] > 0):
            # Potential hydrogen bond
            hbond_potential = 1.0 / (distance + 1e-6)
        features.append(hbond_potential)
        
        # 4. Hydrophobic interaction (both hydrophobic)
        hydro1 = self.get_hydrophobicity(res1['name'])
        hydro2 = self.get_hydrophobicity(res2['name'])
        hydrophobic = 1.0 if hydro1 > 1.0 and hydro2 > 1.0 and distance < 5.0 else 0.0
        features.append(hydrophobic)
        
        return features
    
    def create_pocket_labels(self, residues, ligand_file):
        """Create binary labels for pocket residues (same as before)"""
        try:
            ligand_coords = self.parse_ligand_coords(ligand_file)
            
            if len(ligand_coords) == 0:
                return torch.zeros(len(residues))
            
            labels = []
            for res in residues:
                # Calculate minimum distance between residue and ligand
                res_coords = np.array(res['coords'])
                min_dist = float('inf')
                
                for lig_coord in ligand_coords:
                    distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                    min_dist = min(min_dist, distances.min())
                
                # Label as pocket if any atom within 4Å of ligand
                labels.append(1.0 if min_dist < 4.0 else 0.0)
            
            return torch.tensor(labels, dtype=torch.float)
            
        except Exception as e:
            print(f"Error creating labels: {e}")
            return torch.zeros(len(residues))
    
    def parse_ligand_coords(self, ligand_file):
        """Parse ligand coordinates (same as before)"""
        coords = []
        
        try:
            with open(ligand_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.startswith('ATOM') or (len(line.split()) >= 4 and not line.startswith('#')):
                    parts = line.split()
                    try:
                        if len(parts) >= 4:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            coords.append([x, y, z])
                    except:
                        continue
        
        except Exception as e:
            print(f"Warning: Could not parse ligand file: {e}")
        
        return coords

# Test function
def test_physics_builder():
    """Test the physics-enhanced graph builder"""
    builder = PhysicsEnhancedGraphBuilder(cutoff_distance=8.0)
    
    # Find a sample protein
    sample_protein = "./data/PDBbind/refined-set/1a0q/1a0q_protein.pdb"
    sample_ligand = "./data/PDBbind/refined-set/1a0q/1a0q_ligand.mol2"
    
    if os.path.exists(sample_protein):
        print(f"Testing physics builder on {sample_protein}")
        graph = builder.build_from_pdb(sample_protein, sample_ligand, "1a0q_test")
        
        if graph is not None:
            print(f"✓ Physics graph created successfully!")
            print(f"  Nodes: {graph.num_nodes}")
            print(f"  Edges: {graph.edge_index.shape[1]}")
            print(f"  Node features: {graph.x.shape} (should be [N, 35])")
            print(f"  Edge physics: {graph.edge_physics.shape} (should be [E, 4])")
            print(f"  Pocket residues: {graph.y.sum().item()}/{graph.num_nodes}")
            
            # Save sample graph
            os.makedirs("./data/processed/physics_graphs/samples", exist_ok=True)
            torch.save(graph, "./data/processed/physics_graphs/samples/1a0q_physics.pt")
            print(f"✓ Graph saved to ./data/processed/physics_graphs/samples/1a0q_physics.pt")
            
            return graph
        else:
            print("✗ Failed to create physics graph")
            return None
    else:
        print(f"✗ Sample protein not found")
        return None

if __name__ == "__main__":
    test_physics_builder()
