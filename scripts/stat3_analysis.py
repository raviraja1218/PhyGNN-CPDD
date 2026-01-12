#!/usr/bin/env python3
"""
STAT3 Case Study Analysis
Most important therapeutic target for paper
"""
import os
import sys
import json

# Add pipeline to path
sys.path.append('.')
from scripts.full_pipeline_static import PhyGNNPipeline

def analyze_stat3():
    """Analyze STAT3 transcription factor"""
    print("\n" + "="*70)
    print("STAT3 CASE STUDY ANALYSIS")
    print("="*70)
    
    # Initialize pipeline
    pipeline = PhyGNNPipeline()
    
    if pipeline.model is None:
        print("Error: Pipeline not initialized")
        return
    
    # STAT3 information
    stat3_info = {
        'protein_name': 'Signal Transducer and Activator of Transcription 3',
        'uniprot_id': 'P40763',
        'therapeutic_areas': ['Cancer immunotherapy', 'Autoimmune diseases', 'Inflammation'],
        'druggability_challenge': 'Considered "undruggable" due to flat protein-protein interfaces',
        'known_pockets': 'SH2 domain binding site for phosphorylation',
        'cryptic_potential': 'High - conformational changes may expose new pockets'
    }
    
    print(f"\n📋 STAT3 Information:")
    for key, value in stat3_info.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Try to find STAT3 structure
    stat3_pdb_codes = ['6njs', '6nq0', '1bg1']  # Common STAT3 structures
    
    for pdb_code in stat3_pdb_codes:
        pdb_path = f"./data/PDBbind/refined-set/{pdb_code}/{pdb_code}_protein.pdb"
        
        if os.path.exists(pdb_path):
            print(f"\n🔍 Found STAT3 structure: {pdb_code}")
            
            ligand_path = f"./data/PDBbind/refined-set/{pdb_code}/{pdb_code}_ligand.mol2"
            if not os.path.exists(ligand_path):
                ligand_path = None
            
            # Process STAT3
            results = pipeline.process_protein(pdb_path, ligand_path, f"STAT3_{pdb_code}")
            
            if results:
                # Save STAT3 results
                output_dir = "./experiments/results/phase4/case_studies/STAT3"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save prediction results
                with open(os.path.join(output_dir, 'prediction_analysis.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Create therapeutic relevance analysis
                therapeutic_analysis = {
                    'protein': 'STAT3',
                    'pdb_code': pdb_code,
                    'therapeutic_implications': {
                        'tcell_exhaustion': 'STAT3 drives T-cell exhaustion in cancer',
                        'immunotherapy': 'STAT3 inhibition could enhance CAR-T and checkpoint therapies',
                        'current_status': 'STAT3 considered undruggable due to flat surfaces',
                        'our_finding': f"Identified {results['pocket_residues']} potential pocket residues",
                        'significance': 'Our method may reveal new druggable sites on STAT3'
                    },
                    'prediction_summary': {
                        'total_residues': results['total_residues'],
                        'pocket_residues': results['pocket_residues'],
                        'pocket_percentage': results['pocket_percentage'],
                        'model_confidence': {
                            'min_prob': min(results['probabilities']),
                            'max_prob': max(results['probabilities']),
                            'avg_prob': sum(results['probabilities']) / len(results['probabilities'])
                        }
                    }
                }
                
                with open(os.path.join(output_dir, 'therapeutic_relevance.json'), 'w') as f:
                    json.dump(therapeutic_analysis, f, indent=2)
                
                print(f"\n✅ STAT3 analysis saved to {output_dir}")
                print(f"\n📊 STAT3 Prediction Summary:")
                print(f"   • Total residues: {results['total_residues']}")
                print(f"   • Pocket residues: {results['pocket_residues']} ({results['pocket_percentage']:.1f}%)")
                print(f"   • Processing time: {results['processing_time_seconds']}s")
                
                # Print therapeutic implications
                print(f"\n💊 Therapeutic Implications:")
                for imp_key, imp_value in therapeutic_analysis['therapeutic_implications'].items():
                    print(f"   • {imp_key.replace('_', ' ').title()}: {imp_value}")
                
                break
    
    else:
        print("\n⚠️ No STAT3 structure found in PDBbind dataset")
        print("Creating placeholder analysis for paper...")
        
        # Create placeholder analysis for paper
        placeholder_dir = "./experiments/results/phase4/case_studies/STAT3"
        os.makedirs(placeholder_dir, exist_ok=True)
        
        placeholder = {
            'note': 'STAT3 structure not in PDBbind - using literature data',
            'literature_findings': {
                'known_binding_sites': ['SH2 domain for phosphorylation'],
                'cryptic_pockets': ['Potential pocket near helix αB unwinding'],
                'inhibitors': ['Static (small molecule), SH2 domain inhibitors'],
                'druggability_challenge': 'Flat protein-protein interaction surfaces'
            },
            'our_method_potential': {
                'application': 'Could identify novel cryptic pockets on STAT3',
                'significance': 'Enable targeting of currently "undruggable" proteins',
                'future_work': 'Apply conformational sampling to reveal transient pockets'
            }
        }
        
        with open(os.path.join(placeholder_dir, 'literature_analysis.json'), 'w') as f:
            json.dump(placeholder, f, indent=2)
        
        print(f"\n✅ Placeholder analysis saved to {placeholder_dir}")

def main():
    """Main execution"""
    analyze_stat3()
    
    print(f"\n{'='*70}")
    print("STAT3 CASE STUDY COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
