# PhyGNN-CPDD

**PhyGNN-CPDD** is a physics-informed graph neural network framework for protein pocket detection and druggability analysis.

## Key Contributions
- Hamiltonian-informed GNN for protein pocket detection
- Quantified physics contribution to ML performance
- State-of-the-art results on PDBbind v2020
- Complete reproducible pipeline and analysis

## Repository Structure
- `src/` – core model and training code  
- `scripts/` – execution pipelines and analysis  
- `experiments/` – summarized results (small files only)  
- `paper/` – manuscript, figures, and tables  
- `data/` – instructions only (datasets not included)

## Dataset
This project uses **PDBbind v2020 (Refined Set)**.  
Due to licensing and size, datasets are **not included**.

See `data/README.md` for download instructions.

## Reproducibility
```bash
conda env create -f environment.yml
conda activate phygnn-cpdd
python scripts/full_pipeline_static_fixed.py


---

# ✅ STEP 2: Create `LICENSE` (MIT — required for research repos)

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Ravi Raja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
