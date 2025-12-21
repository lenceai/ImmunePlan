#!/bin/bash

set -euo pipefail

echo "========================================="
echo "AUTOIMMUNE LLM PROJECT SETUP"
echo "========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Python check
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found in PATH"
    echo "Please install Python 3.10+ (recommended: 3.11+)."
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# Optional venv (recommended). Set VENV_DIR to override, or SKIP_VENV=1 to disable.
VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV="${USE_VENV:-0}"
if [ "$USE_VENV" = "1" ]; then
    if [ -d "$VENV_DIR" ] && [ "${RECREATE_VENV:-0}" = "1" ]; then
        echo "Recreating venv (RECREATE_VENV=1): removing $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating venv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR" || {
            echo "ERROR: failed to create venv."
            echo "On Ubuntu/Debian you may need: sudo apt-get install python3-venv"
            exit 1
        }
    fi

    echo "Activating venv: $VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
else
    echo "Using current Python environment (USE_VENV=0)."
    echo "  If you want an isolated venv instead, run: USE_VENV=1 ./setup.sh"
fi

# Install PyTorch 2.7.1 with CUDA support via pip (conda may not have latest version)
echo ""
if [ "${SKIP_TORCH:-0}" = "1" ]; then
    echo "Skipping PyTorch install (SKIP_TORCH=1)."
else
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    echo "Installing PyTorch 2.7.1 from: $TORCH_INDEX_URL"
    python -m pip install --upgrade pip
    python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$TORCH_INDEX_URL"
fi

# Install any additional requirements
echo ""
echo "Installing additional requirements..."
python -m pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data results models checkpoints logs docs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# PubMed API Configuration (REQUIRED)
PUBMED_EMAIL=your.email@example.com

# Weights & Biases (Optional)
WANDB_API_KEY=
WANDB_PROJECT=autoimmune-llm

# Model Configuration
MODEL_NAME=nvidia/Nemotron-Cascade-8B-Thinking
MAX_SEQ_LENGTH=1024

# Training Configuration
QLORA_EPOCHS=3
QLORA_BATCH_SIZE=1
QLORA_GRAD_ACCUM=16
QLORA_LR=2e-4

FULL_EPOCHS=3
FULL_BATCH_SIZE=1
FULL_LR=5e-5

# Output Directories
DATA_DIR=./data
RESULTS_DIR=./results
MODELS_DIR=./models
EOF
    echo "Please edit .env and add your email for PubMed API"
fi

# Verify CUDA availability
echo ""
echo "Verifying CUDA availability..."
python - <<'PY'
try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
except Exception as e:
    print("WARNING: Could not import torch to verify CUDA.")
    print(f"  {e}")
PY

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your email for PubMed"
echo "2. (Recommended) Use a venv: USE_VENV=1 ./setup.sh  # then: source ${VENV_DIR}/bin/activate"
echo "3. Run: ./run_all.sh"

