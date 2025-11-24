#!/bin/bash

echo "========================================="
echo "AUTOIMMUNE LLM PROJECT SETUP"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^Plan "; then
    echo "WARNING: Conda environment 'Plan' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n Plan -y
    else
        echo "Using existing environment. Activate it with: conda activate Plan"
        exit 0
    fi
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi

# Activate environment
echo ""
echo "Activating conda environment 'Plan'..."
eval "$(conda shell.bash hook)"
conda activate Plan

# Install PyTorch 2.7.1 with CUDA support via pip (conda may not have latest version)
echo ""
echo "Installing PyTorch 2.7.1 with CUDA 12.8..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install any additional requirements
echo ""
echo "Installing additional requirements..."
pip install -r requirements.txt

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
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B
MAX_SEQ_LENGTH=2048

# Training Configuration
QLORA_EPOCHS=3
QLORA_BATCH_SIZE=2
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
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your email for PubMed"
echo "2. Activate environment: conda activate Plan"
echo "3. Run: ./run_all.sh"

