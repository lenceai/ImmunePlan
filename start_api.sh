#!/bin/bash
cd ~/ImmunePlan
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate Plan
export PORT=5000
export HOST=0.0.0.0
python api.py
