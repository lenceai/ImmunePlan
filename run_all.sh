#!/bin/bash

echo "========================================="
echo "AUTOIMMUNE LLM FINE-TUNING PIPELINE"
echo "========================================="
echo ""

# Always activate conda environment 'Plan' before running anything
echo "Activating conda environment 'Plan'..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate Plan environment
if conda env list | grep -q "^Plan "; then
    conda activate Plan
    echo "✓ Conda environment 'Plan' activated"
else
    echo "ERROR: Conda environment 'Plan' not found."
    echo "Please run ./setup.sh first to create the environment."
    exit 1
fi

# Verify environment is activated
if [ "$CONDA_DEFAULT_ENV" != "Plan" ]; then
    echo "ERROR: Failed to activate conda environment 'Plan'"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Set script directory
SCRIPT_DIR="scripts"

# Function to run a script
run_script() {
    local script_num=$1
    local script_name=$2
    local script_path="$SCRIPT_DIR/${script_num}_${script_name}.py"
    
    echo ""
    echo "========================================="
    echo "Running: $script_name"
    echo "========================================="
    echo ""
    
    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script not found: $script_path"
        return 1
    fi
    
    python "$script_path"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Script failed: $script_name"
        echo "Pipeline stopped."
        exit 1
    fi
}

# Run pipeline steps
echo "Starting pipeline execution..."
echo ""

# Step 1: Download and test
run_script "1" "download_and_test"

# Step 2: Baseline benchmark
run_script "2" "autoimmune_questions"

# Step 3: Download papers
run_script "3" "download_papers"

# Step 4: QLoRA fine-tuning
run_script "4" "qlora_finetune"

# Step 5: Test QLoRA
run_script "5" "test_qlora"

# Step 6: Full fine-tuning (optional - will prompt if insufficient VRAM)
echo ""
echo "========================================="
echo "Step 6: Full Fine-Tuning (Optional)"
echo "========================================="
echo ""
echo "This step requires significant VRAM (40GB+)."
read -p "Run full fine-tuning? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_script "6" "full_finetune"
    
    # Step 7: Test full model (only if step 6 completed)
    if [ $? -eq 0 ]; then
        run_script "7" "test_full_model"
    fi
else
    echo "Skipping full fine-tuning."
fi

# Step 8: Score and compare
run_script "8" "score_results"

echo ""
echo "========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================="
echo ""
echo "Results are available in:"
echo "  - results/model_comparison.csv"
echo "  - results/final_report.txt"
echo ""

