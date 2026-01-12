"""
PDBbind dataset loader for PhyGNN-CPDD with GPU support.
"""
import os
import pandas as pd
import torch
from typing import List, Dict, Optional
from pathlib import Path
import warnings


class PDBbindLoader:
    """Loader for PDBbind dataset with GPU support."""
    
    def __init__(self, data_dir: str = "./data/PDBbind/refined-set", device: str = 'cuda'):
        """
        Initialize loader.
        
        Args:
            data_dir: Path to PDBbind refined set directory
            device: 'cuda' or 'cpu'
        """
        self.data_dir = Path(data_dir)
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Check if directory exists
        if not self.data_dir.exists():
            # Try alternative paths
            alt_path = Path("./data/PDBbind")
            if alt_path.exists():
                # Find refined-set inside
                refined_dirs = list(alt_path.glob("*/"))
                if refined_dirs:
                    self.data_dir = refined_dirs[0]
                    print(f"Found dataset at: {self.data_dir}")
                else:
                    raise FileNotFoundError(f"No dataset found at {data_dir}")
            else:
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        print(f"Initialized PDBbindLoader:")
        print(f"  - Dataset: {self.data_dir}")
        print(f"  - Device: {self.device}")
        print(f"  - GPU available: {torch.cuda.is_available()}")
        
    def list_complexes(self) -> List[str]:
        """List all PDB IDs in the dataset."""
        complexes = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if it has required files
                protein_files = list(item.glob("*protein*.pdb"))
                ligand_files = list(item.glob("*ligand*"))
                if protein_files and ligand_files:
                    complexes.append(item.name)
        
        print(f"Found {len(complexes)} valid complexes")
        return complexes
    
    def load_complex(self, pdb_id: str, to_tensor: bool = False) -> Dict:
        """
        Load a single protein-ligand complex.
        
        Args:
            pdb_id: PDB ID (e.g., '1a0q')
            to_tensor: If True, return torch tensors
            
        Returns:
            Dictionary containing protein, ligand, and metadata
        """
        complex_dir = self.data_dir / pdb_id
        
        if not complex_dir.exists():
            raise FileNotFoundError(f"Complex directory not found: {pdb_id}")
        
        # Find protein file (could be .pdb or .ent)
        protein_files = list(complex_dir.glob("*protein*.pdb")) + \
                       list(complex_dir.glob("*.pdb"))
        
        # Find ligand file (could be .mol2, .sdf, .pdb)
        ligand_files = list(complex_dir.glob("*ligand*.mol2")) + \
                      list(complex_dir.glob("*ligand*.sdf")) + \
                      list(complex_dir.glob("*ligand*.pdb"))
        
        if not protein_files:
            # Try to find any .pdb file
            all_pdbs = list(complex_dir.glob("*.pdb"))
            if all_pdbs:
                # Assume first is protein, rest might be ligand
                protein_files = [all_pdbs[0]]
                ligand_files = all_pdbs[1:] if len(all_pdbs) > 1 else []
        
        if not protein_files:
            raise FileNotFoundError(f"No protein file found for {pdb_id}")
        
        result = {
            'pdb_id': pdb_id,
            'protein_path': str(protein_files[0]),
            'ligand_path': str(ligand_files[0]) if ligand_files else None,
            'complex_dir': str(complex_dir),
            'has_ligand': bool(ligand_files),
            'device': self.device
        }
        
        if to_tensor:
            # Convert to tensors (placeholder - will implement later)
            result['protein_tensor'] = torch.randn(100, 3, device=self.device)  # Placeholder
            if ligand_files:
                result['ligand_tensor'] = torch.randn(50, 3, device=self.device)  # Placeholder
        
        return result
    
    def load_all(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load metadata for all complexes.
        
        Args:
            limit: Optional limit on number of complexes
            
        Returns:
            DataFrame with complex metadata
        """
        complexes = self.list_complexes()
        if limit:
            complexes = complexes[:limit]
        
        data = []
        for i, pdb_id in enumerate(complexes):
            try:
                complex_info = self.load_complex(pdb_id)
                data.append(complex_info)
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Loaded {i + 1}/{len(complexes)} complexes")
                    
            except Exception as e:
                print(f"Warning: Failed to load {pdb_id}: {e}")
        
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)}/{len(complexes)} complexes")
        return df
    
    def get_dataset_stats(self) -> Dict:
        """Get basic statistics about the dataset."""
        complexes = self.list_complexes()
        
        # Load first 100 for detailed stats
        sample_size = min(100, len(complexes))
        sample_complexes = complexes[:sample_size]
        
        protein_sizes = []
        has_ligand_count = 0
        
        for pdb_id in sample_complexes:
            try:
                complex_info = self.load_complex(pdb_id)
                # Count protein atoms (simplified - just file size)
                protein_path = Path(complex_info['protein_path'])
                if protein_path.exists():
                    with open(protein_path, 'r') as f:
                        lines = f.readlines()
                        protein_sizes.append(len([l for l in lines if l.startswith('ATOM')]))
                
                if complex_info['has_ligand']:
                    has_ligand_count += 1
            except:
                pass
        
        import numpy as np
        
        stats = {
            'total_complexes': len(complexes),
            'sample_analyzed': sample_size,
            'avg_protein_atoms': np.mean(protein_sizes) if protein_sizes else 0,
            'min_protein_atoms': np.min(protein_sizes) if protein_sizes else 0,
            'max_protein_atoms': np.max(protein_sizes) if protein_sizes else 0,
            'complexes_with_ligand': has_ligand_count,
            'ligand_percentage': (has_ligand_count / sample_size * 100) if sample_size > 0 else 0,
            'sample_pdbs': complexes[:5],
            'device': self.device
        }
        
        return stats
    
    def create_validation_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create train/val/test splits."""
        import numpy as np
        
        complexes = self.list_complexes()
        np.random.seed(seed)
        np.random.shuffle(complexes)
        
        n = len(complexes)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            'train': complexes[:train_end],
            'val': complexes[train_end:val_end],
            'test': complexes[val_end:]
        }


if __name__ == "__main__":
    # Test the loader with GPU
    print("Testing PDBbindLoader with GPU support...")
    loader = PDBbindLoader(device='cuda')
    
    stats = loader.get_dataset_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if key != 'sample_pdbs':
            print(f"  {key}: {value}")
    
    print(f"  sample_pdbs: {stats['sample_pdbs']}")
    
    # Test splits
    splits = loader.create_validation_split()
    print(f"\nDataset Splits:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} complexes")
