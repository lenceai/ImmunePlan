#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Optional venv activation
VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV="${USE_VENV:-0}"
if [ "$USE_VENV" = "1" ]; then
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
  else
    echo "ERROR: USE_VENV=1 but venv not found at ${VENV_DIR}."
    echo "  Run: USE_VENV=1 ./setup.sh"
    exit 1
  fi
fi
export PORT=5000
export HOST=0.0.0.0
python api.py
