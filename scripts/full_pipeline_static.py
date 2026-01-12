#!/usr/bin/env python3
"""
Complete PhyGNN Pipeline: PDB -> Physics Graph -> Prediction
Single file that integrates all working components
"""
import os
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path

# Add our modules to path
sys.path.append('./src/models')

# Import our working components
try:
    from physics_enhanced_builder import PhysicsEnhancedGraphBuilder
    from hamiltonian_gnn_phase2c import HamiltonianGNN
    print("✓ Successfully imported all components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

class PhyGNNPipeline:
    """Complete pipeline for protein pocket prediction"""
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize pipeline with pretrained model
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize graph builder
        self.graph_builder = PhysicsEnhancedGraphBuilder(cutoff_distance=8.0)
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Statistics
        self.stats = {
            'total_proteins_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_processing_time': 0,
            'timings': []
        }
    
    def load_model(self, model_path):
        """Load trained Hamiltonian GNN"""
        if model_path is None:
            # Use our best model from Phase 2B
            model_path = "./experiments/results/phase2b/week2/training_fixed/hamgnn_best.pt"
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return None
        
        try:
            # First, create a dummy model to get architecture
            # We need to know input dimension
            sample_graph = self.load_sample_graph()
            if sample_graph is None:
                print("Error: Could not load sample graph to determine input dimensions")
                return None
            
            input_dim = sample_graph.x.shape[1]
            print(f"Model input dimension: {input_dim}")
            
            # Create model with same architecture
            model = HamiltonianGNN(
                input_dim=input_dim,
                hidden_dim=128,  # From Phase 2B
                lambda_physics=0.0001  # Optimal from Phase 2B
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"✓ Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def load_sample_graph(self):
        """Load a sample graph to determine input dimensions"""
        sample_path = "./data/processed/physics_graphs/train/1a0q_graph.pt"
        if os.path.exists(sample_path):
            try:
                return torch.load(sample_path, weights_only=True)
            except:
                return None
        return None
    
    def process_protein(self, pdb_path, ligand_path=None, protein_id=None):
        """
        Process a single protein from PDB to predictions
        
        Args:
            pdb_path: Path to protein PDB file
            ligand_path: Path to ligand file (optional)
            protein_id: Identifier for the protein
            
        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        
        if protein_id is None:
            protein_id = os.path.basename(pdb_path).replace('_protein.pdb', '')
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {protein_id}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Build graph from PDB
            print("1. Building physics-enhanced graph...")
            graph = self.graph_builder.build_from_pdb(pdb_path, ligand_path, protein_id)
            
            if graph is None:
                print("✗ Failed to build graph")
                self.stats['failed_predictions'] += 1
                return None
            
            print(f"   ✓ Graph built: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
            
            # Step 2: Prepare graph for model
            print("2. Preparing graph for prediction...")
            graph = graph.to(self.device)
            
            # Step 3: Make prediction
            print("3. Running Hamiltonian GNN prediction...")
            with torch.no_grad():
                logits = self.model(graph)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
            
            # Step 4: Extract results
            print("4. Extracting pocket predictions...")
            pred_array = predictions.cpu().numpy().flatten()
            prob_array = probabilities.cpu().numpy().flatten()
            
            # Get pocket residues (indices where prediction == 1)
            pocket_indices = np.where(pred_array == 1)[0].tolist()
            
            # Calculate metrics
            total_residues = len(pred_array)
            pocket_count = len(pocket_indices)
            pocket_percentage = (pocket_count / total_residues * 100) if total_residues > 0 else 0
            
            # Step 5: Create results dictionary
            processing_time = time.time() - start_time
            self.stats['timings'].append(processing_time)
            self.stats['total_proteins_processed'] += 1
            self.stats['successful_predictions'] += 1
            self.stats['avg_processing_time'] = np.mean(self.stats['timings'])
            
            results = {
                'protein_id': protein_id,
                'total_residues': total_residues,
                'pocket_residues': pocket_count,
                'pocket_percentage': round(pocket_percentage, 2),
                'pocket_indices': pocket_indices,
                'processing_time_seconds': round(processing_time, 2),
                'probabilities': prob_array.tolist(),
                'graph_metadata': {
                    'num_nodes': graph.num_nodes,
                    'num_edges': graph.edge_index.shape[1],
                    'has_physics_features': hasattr(graph, 'physics_features')
                },
                'success': True
            }
            
            print(f"\n📊 RESULTS for {protein_id}:")
            print(f"   • Total residues: {total_residues}")
            print(f"   • Pocket residues: {pocket_count} ({pocket_percentage:.1f}%)")
            print(f"   • Processing time: {processing_time:.2f}s")
            print(f"   • Model confidence range: [{prob_array.min():.3f}, {prob_array.max():.3f}]")
            
            return results
            
        except Exception as e:
            print(f"✗ Error processing {protein_id}: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_predictions'] += 1
            return None
    
    def save_predictions(self, results, output_dir):
        """Save prediction results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{results['protein_id']}_prediction.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Predictions saved to {output_path}")
        return output_path
    
    def run_test_suite(self, test_proteins=10):
        """
        Run pipeline on test proteins
        
        Args:
            test_proteins: Number of test proteins to process
        
        Returns:
            Test results summary
        """
        print(f"\n{'='*60}")
        print(f"RUNNING TEST SUITE ({test_proteins} proteins)")
        print(f"{'='*60}")
        
        # Get list of available proteins
        pdb_dir = "./data/PDBbind/refined-set"
        protein_dirs = []
        
        for item in os.listdir(pdb_dir):
            dir_path = os.path.join(pdb_dir, item)
            if os.path.isdir(dir_path):
                protein_dirs.append(item)
        
        # Take first N proteins
        test_dirs = protein_dirs[:test_proteins]
        
        all_results = []
        
        for protein_id in test_dirs:
            pdb_path = os.path.join(pdb_dir, protein_id, f"{protein_id}_protein.pdb")
            ligand_path = os.path.join(pdb_dir, protein_id, f"{protein_id}_ligand.mol2")
            
            if not os.path.exists(pdb_path):
                print(f"✗ Missing PDB file for {protein_id}")
                continue
            
            # Try alternative ligand formats
            if not os.path.exists(ligand_path):
                ligand_path = os.path.join(pdb_dir, protein_id, f"{protein_id}_ligand.sdf")
            
            results = self.process_protein(pdb_path, ligand_path, protein_id)
            
            if results:
                all_results.append(results)
                
                # Save individual results
                output_dir = "./experiments/results/phase4/pipeline/sample_predictions"
                self.save_predictions(results, output_dir)
        
        # Save summary statistics
        self.save_test_summary(all_results)
        
        return all_results
    
    def save_test_summary(self, results):
        """Save test suite summary"""
        summary = {
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_proteins_tested': len(results),
            'success_rate': len(results) / self.stats['total_proteins_processed'] if self.stats['total_proteins_processed'] > 0 else 0,
            'statistics': self.stats,
            'per_protein_results': [
                {
                    'protein_id': r['protein_id'],
                    'pocket_residues': r['pocket_residues'],
                    'processing_time': r['processing_time_seconds']
                } for r in results
            ]
        }
        
        output_path = "./experiments/results/phase4/pipeline/end_to_end_test_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total proteins tested: {len(results)}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average processing time: {self.stats['avg_processing_time']:.2f}s")
        print(f"Total successful predictions: {self.stats['successful_predictions']}")
        print(f"Total failed predictions: {self.stats['failed_predictions']}")
        print(f"Summary saved to: {output_path}")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("PHYGNN COMPLETE PIPELINE - END-TO-END TEST")
    print("="*70)
    
    # Initialize pipeline
    pipeline = PhyGNNPipeline()
    
    if pipeline.model is None:
        print("✗ Failed to initialize pipeline. Exiting.")
        return
    
    # Run test suite
    results = pipeline.run_test_suite(test_proteins=10)
    
    # Save timing analysis
    if pipeline.stats['timings']:
        timings_path = "./experiments/results/phase4/pipeline/inference_timings.csv"
        with open(timings_path, 'w') as f:
            f.write("protein_index,time_seconds\n")
            for i, t in enumerate(pipeline.stats['timings']):
                f.write(f"{i},{t}\n")
        print(f"\n✓ Timing data saved to {timings_path}")
    
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
