#!/bin/bash
set -euo pipefail

echo "========================================="
echo "  IMMUNEPLAN PIPELINE"
echo "  Building Reliable AI Systems"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP="${SKIP:-}"

run_step() {
    local num=$1
    local name=$2

    if echo "$SKIP" | grep -q "$num"; then
        echo ""
        echo "--- Skipping step $num: $name ---"
        return 0
    fi

    echo ""
    echo "========================================="
    echo "  Running: Step $num — $name"
    echo "========================================="

    python3 "pipeline/${num}_${name}.py" || {
        echo "Step $num failed."
        # GPU steps (02, 05, 06) are optional
        case "$num" in
            02|05|06) echo "  (GPU step — continuing)" ;;
            *) exit 1 ;;
        esac
    }
}

run_step 01 "setup"
run_step 02 "baseline"
run_step 03 "collect_data"
run_step 04 "build_rag"
run_step 05 "finetune"
run_step 06 "test_model"
run_step 07 "build_agent"
run_step 08 "evaluate"
run_step 09 "safety"
run_step 10 "deploy"

echo ""
echo "========================================="
echo "  PIPELINE COMPLETE"
echo "========================================="
echo "Results: results/"
echo "API:     python api.py"
echo ""
