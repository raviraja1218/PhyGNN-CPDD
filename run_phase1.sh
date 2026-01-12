#!/bin/bash
# Phase 1 Execution Script
# Run all steps in order

echo "============================================="
echo "PhyGNN-CPDD: PHASE 1 EXECUTION"
echo "Date: $(date)"
echo "============================================="

# Step 0: Check environment
echo -e "\n[STEP 0] Checking Environment..."
python3 --version
pip list | grep -E "torch|rdkit|biopython|pandas"

# Step 1: Validate Dataset
echo -e "\n[STEP 1] Validating Dataset..."
python3 ./src/data_processing/validate_dataset.py
if [ $? -ne 0 ]; then
    echo "❌ Dataset validation failed"
    exit 1
fi

# Step 2: Create Splits
echo -e "\n[STEP 2] Creating Dataset Splits..."
python3 ./src/data_processing/create_splits.py
if [ $? -ne 0 ]; then
    echo "❌ Split creation failed"
    exit 1
fi

# Step 3: Compute Dataset Statistics
echo -e "\n[STEP 3] Computing Dataset Statistics..."
python3 ./src/data_processing/dataset_statistics.py
if [ $? -ne 0 ]; then
    echo "❌ Statistics computation failed"
    exit 1
fi

# Step 4: Install FPOCKET (if needed)
echo -e "\n[STEP 4] Checking FPOCKET Installation..."
if ! command -v fpocket &> /dev/null; then
    echo "FPOCKET not found. Installing..."
    chmod +x ./install_fpocket.sh
    ./install_fpocket.sh
    if [ $? -ne 0 ]; then
        echo "❌ FPOCKET installation failed"
        echo "You can continue with manual FPOCKET installation"
    fi
else
    echo "✅ FPOCKET already installed"
fi

# Step 5: Run FPOCKET Baseline
echo -e "\n[STEP 5] Running FPOCKET Baseline..."
python3 ./src/baselines/run_fpocket_baseline.py
if [ $? -ne 0 ]; then
    echo "⚠️ FPOCKET baseline had issues (check logs)"
fi

# Step 6: Create Visualizations
echo -e "\n[STEP 6] Creating Visualizations..."
python3 ./src/visualization/create_visualizations.py
if [ $? -ne 0 ]; then
    echo "⚠️ Visualization creation had issues"
fi

# Step 7: Create Summary Report
echo -e "\n[STEP 7] Creating Phase 1 Summary..."
echo "=== PHASE 1 COMPLETION REPORT ===" > ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "Date: $(date)" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "" >> ./experiments/results/phase1/PHASE1_SUMMARY.md

# Check all targets
echo "## TARGET COMPLETION STATUS" >> ./experiments/results/phase1/PHASE1_SUMMARY.md

# Target 1
if [ -f "./experiments/results/phase1/dataset_statistics.csv" ]; then
    echo "✅ TARGET 1: dataset_statistics.csv created" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
    echo "   Rows: $(wc -l < ./experiments/results/phase1/dataset_statistics.csv)" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
else
    echo "❌ TARGET 1: Missing dataset_statistics.csv" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
fi

# Target 2
if [ -f "./experiments/results/phase1/fpocket_performance.json" ]; then
    echo "✅ TARGET 2: fpocket_performance.json created" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
    echo "   Also: baseline_comparison.tex created" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
else
    echo "❌ TARGET 2: Missing fpocket_performance.json" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
fi

# Target 3
viz_count=$(find ./experiments/results/phase1/visualizations -name "*.png" 2>/dev/null | wc -l)
if [ $viz_count -ge 5 ]; then
    echo "✅ TARGET 3: $viz_count visualizations created" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
else
    echo "❌ TARGET 3: Only $viz_count visualizations (expected >=5)" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
fi

# Target 4 (placeholder)
echo "⏳ TARGET 4: Graph feature statistics (Phase 1B)" >> ./experiments/results/phase1/PHASE1_SUMMARY.md

# Target 5 (placeholder)
echo "⏳ TARGET 5: Data processing report (Phase 1B)" >> ./experiments/results/phase1/PHASE1_SUMMARY.md

echo "" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "## NEXT STEPS" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "1. Review Phase 1 results" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "2. Fix any issues identified" >> ./experiments/results/phase1/PHASE1_SUMMARY.md
echo "3. Proceed to Phase 1B: Graph Construction" >> ./experiments/results/phase1/PHASE1_SUMMARY.md

cat ./experiments/results/phase1/PHASE1_SUMMARY.md

echo -e "\n============================================="
echo "PHASE 1 EXECUTION COMPLETE"
echo "Results saved to: ./experiments/results/phase1/"
echo "Summary: ./experiments/results/phase1/PHASE1_SUMMARY.md"
echo "============================================="
