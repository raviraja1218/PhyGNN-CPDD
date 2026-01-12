"""
Enhanced Feature Extraction for Proteins
"""
import torch
import numpy as np
import os

class EnhancedFeatureBuilder:
    def __init__(self):
        # Hydrophobicity scale (Kyte-Doolittle)
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
            'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
            'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }
        
        # Volume (in Å³)
        self.volume = {
            'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
            'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
            'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
            'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
        }
        
        # Charge at pH 7
        self.charge = {
            'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
            'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.5, 'ILE': 0,
            'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
            'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
        }
        
        # Secondary structure propensity (helix, sheet, coil)
        self.ss_propensity = {
            'ALA': [1.45, 0.97, 0.58], 'ARG': [0.96, 0.99, 1.05],
            'ASN': [0.67, 0.89, 1.44], 'ASP': [1.24, 0.72, 1.04],
            'CYS': [0.77, 1.30, 0.93], 'GLN': [1.17, 1.23, 0.60],
            'GLU': [1.53, 0.26, 1.21], 'GLY': [0.53, 0.81, 1.66],
            'HIS': [1.00, 0.87, 1.13], 'ILE': [1.00, 1.60, 0.40],
            'LEU': [1.34, 1.22, 0.44], 'LYS': [1.18, 0.90, 0.92],
            'MET': [1.20, 1.67, 0.13], 'PHE': [1.12, 1.28, 0.60],
            'PRO': [0.57, 0.55, 1.88], 'SER': [0.77, 0.75, 1.48],
            'THR': [0.83, 1.19, 0.98], 'TRP': [1.08, 1.37, 0.55],
            'TYR': [0.69, 1.47, 0.84], 'VAL': [1.14, 1.65, 0.21]
        }
    
    def get_residue_features(self, residue_name):
        """Get comprehensive residue features"""
        if residue_name not in self.hydrophobicity:
            # Unknown residue, return zeros
            return [0] * 28
        
        features = []
        
        # 1. One-hot encoding (20 dim)
        one_hot = np.zeros(20)
        aa_list = list(self.hydrophobicity.keys())
        if residue_name in aa_list:
            one_hot[aa_list.index(residue_name)] = 1.0
        features.extend(one_hot)
        
        # 2. Physicochemical properties (6 dim)
        features.append(self.hydrophobicity.get(residue_name, 0))
        features.append(self.volume.get(residue_name, 0) / 200.0)  # Normalized
        features.append(self.charge.get(residue_name, 0))
        
        # 3. Secondary structure propensity (3 dim)
        features.extend(self.ss_propensity.get(residue_name, [0, 0, 0]))
        
        # 4. Residue groups (7 dim)
        # Aliphatic, Aromatic, Polar, Charged, Small, Hydrophobic, Special
        aliphatic = 1.0 if residue_name in ['ALA', 'ILE', 'LEU', 'VAL'] else 0.0
        aromatic = 1.0 if residue_name in ['PHE', 'TYR', 'TRP', 'HIS'] else 0.0
        polar = 1.0 if residue_name in ['ASN', 'GLN', 'SER', 'THR'] else 0.0
        charged = 1.0 if residue_name in ['ARG', 'LYS', 'ASP', 'GLU'] else 0.0
        small = 1.0 if residue_name in ['ALA', 'GLY', 'SER'] else 0.0
        hydrophobic = 1.0 if self.hydrophobicity.get(residue_name, 0) > 0 else 0.0
        special = 1.0 if residue_name in ['CYS', 'PRO', 'MET'] else 0.0
        
        features.extend([aliphatic, aromatic, polar, charged, small, hydrophobic, special])
        
        return features
    
    def get_positional_features(self, residue_index, total_residues):
        """Get positional features"""
        # Normalized position
        norm_pos = residue_index / total_residues
        
        # Position encoding (sine/cosine)
        pos_encoding = [
            np.sin(norm_pos * 2 * np.pi),
            np.cos(norm_pos * 2 * np.pi),
            np.sin(norm_pos * 4 * np.pi),
            np.cos(norm_pos * 4 * np.pi)
        ]
        
        # Position quartile (one-hot)
        quartile = np.zeros(4)
        quartile_idx = min(3, int(norm_pos * 4))
        quartile[quartile_idx] = 1.0
        
        return [norm_pos] + pos_encoding + list(quartile)

def test_enhanced_features():
    """Test the enhanced feature builder"""
    builder = EnhancedFeatureBuilder()
    
    # Test with a few residues
    test_residues = ['ALA', 'ARG', 'ASP', 'PHE', 'GLY']
    
    print("Enhanced Feature Test:")
    print("="*50)
    
    for res in test_residues:
        features = builder.get_residue_features(res)
        print(f"{res}: {len(features)} features")
        print(f"  Hydrophobicity: {builder.hydrophobicity.get(res, 'N/A')}")
        print(f"  Charge: {builder.charge.get(res, 'N/A')}")
        print(f"  Volume: {builder.volume.get(res, 'N/A')} Å³")
        
        # Test positional features
        pos_features = builder.get_positional_features(10, 100)
        print(f"  Positional features: {len(pos_features)} dim")
        print()

if __name__ == "__main__":
    test_enhanced_features()
