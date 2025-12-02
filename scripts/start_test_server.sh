#!/bin/bash
# Usage: ./start_test_server.sh
# 
# This script launches a lightweight OpenAI-compatible API server.
# It uses transformers + FastAPI which is compatible with:
# - Steam Deck (AMD GPU / CPU)
# - Laptops (CPU / Integrated Graphics)
# - NVIDIA GPUs (with CUDA)
# - Conda environments (no compilation needed)
#
# It uses the standard DeepSeek-R1-Distill-Qwen-1.5B model with 4-bit quantization.

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "================================================================"
echo "STARTING TEST SERVER (Transformers + FastAPI)"
echo "================================================================"
echo "Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
echo "Quantization: 4-bit (auto-fallback to FP16 if needed)"
echo "----------------------------------------------------------------"

# Check for required packages
if ! python3 -c "import transformers" &> /dev/null; then
    echo "Installing required packages..."
    pip install transformers torch accelerate bitsandbytes fastapi uvicorn --no-cache-dir
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================"
echo "Launching Server..."
echo "API URL: http://localhost:8000/v1"
echo "Docs:    http://localhost:8000/docs"
echo "================================================================"

# Start the Python server
python3 "$SCRIPT_DIR/test_server.py"
