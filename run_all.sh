#!/bin/bash

set -euo pipefail

echo "========================================="
echo "AUTOIMMUNE LLM FINE-TUNING PIPELINE"
echo "========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Auto-activate venv if present (recommended). Set VENV_DIR to override.
VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV="${USE_VENV:-0}"
if [ "$USE_VENV" = "1" ]; then
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        echo "Activating venv: ${VENV_DIR}"
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
    else
        echo "ERROR: USE_VENV=1 but venv not found at ${VENV_DIR}."
        echo "  Run: USE_VENV=1 ./setup.sh"
        exit 1
    fi
else
    echo "Using current Python environment (USE_VENV=0)."
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Please run ./setup.sh (it will create .env) or create it manually."
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

# Step 4: DoRA fine-tuning (with 4-bit quantization)
run_script "4" "finetune"

# Step 5: Test fine-tuned model (DoRA or QLoRA)
run_script "5" "test_qlora"

# Step 6: Full fine-tuning (optional - will prompt if insufficient VRAM)
echo ""
echo "========================================="
echo "Step 6: Full Fine-Tuning (Optional)"
echo "========================================="
echo ""
echo "This step requires significant VRAM (40GB+)."
if [ "${RUN_FULL_FINETUNE:-0}" = "1" ]; then
    echo "RUN_FULL_FINETUNE=1 set; running full fine-tuning."
    run_script "6" "full_finetune"
    run_script "7" "test_full_model"
else
    echo "Skipping full fine-tuning. (Set RUN_FULL_FINETUNE=1 to enable.)"
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

