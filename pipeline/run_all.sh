#!/bin/bash
set -euo pipefail

# This script is now a wrapper for the main Python pipeline orchestrator.
# run_pipeline.py handles step ordering, GPU isolation, and multi-GPU torchrun.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting pipeline via run_pipeline.py..."
python pipeline/run_pipeline.py "$@"
