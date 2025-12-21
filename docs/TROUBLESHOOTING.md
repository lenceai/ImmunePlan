# Troubleshooting Guide

## Common Issues and Solutions

### Setup Issues

#### Problem: CUDA not available
**Symptoms:** Warning message about CUDA not being available

**Solutions:**
1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```
2. Install CUDA toolkit if missing
3. Reinstall PyTorch 2.7.1 with CUDA support:
   ```bash
   pip install torch==2.7.1 torchvision==0.20.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   Or for CUDA 11.8:
   ```bash
   pip install torch==2.7.1 torchvision==0.20.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
   ```

#### Problem: Python environment not activating / packages not found
**Symptoms:** `ModuleNotFoundError` (e.g. `transformers`), or scripts run with the wrong Python

**Solutions:**
1. If you are using a venv, activate it:
   ```bash
   source .venv/bin/activate
   ```
   (Or rerun setup with `USE_VENV=1 ./setup.sh`.)

2. Verify you’re using the expected Python and that dependencies are installed:
   ```bash
   which python
   python -c "import transformers; print(transformers.__version__)"
   ```

3. Reinstall dependencies if needed:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. If you want to recreate the venv from scratch:
   ```bash
   USE_VENV=1 RECREATE_VENV=1 ./setup.sh
   ```

---

### Model Loading Issues

#### Problem: Out of Memory (OOM) during model loading
**Symptoms:** CUDA out of memory error

**Solutions:**
1. Ensure 4-bit quantization is enabled (default)
2. Close other GPU processes:
   ```bash
   nvidia-smi
   # Kill processes using GPU
   ```
3. Reduce `MAX_SEQ_LENGTH` in `.env`
4. Use CPU offloading if available

#### Problem: Model download fails
**Symptoms:** Connection timeout or download error

**Solutions:**
1. Check internet connection
2. Use Hugging Face mirror if in restricted region
3. Set Hugging Face token if required:
   ```bash
   export HF_TOKEN=your_token
   ```
4. Try downloading manually:
   ```python
   from transformers import AutoModelForCausalLM
   AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
   ```

---

### Training Issues

#### Problem: Out of Memory during QLoRA training
**Symptoms:** CUDA OOM error during training

**Solutions:**
1. Reduce batch size in `.env`:
   ```bash
   QLORA_BATCH_SIZE=1
   ```
2. Increase gradient accumulation steps:
   ```bash
   # Already set to 8, can increase further
   ```
3. Reduce sequence length:
   ```bash
   MAX_SEQ_LENGTH=1024
   ```
4. Enable gradient checkpointing (already enabled)

#### Problem: Training loss not decreasing
**Symptoms:** Loss stays constant or increases

**Solutions:**
1. Reduce learning rate:
   ```bash
   QLORA_LR=1e-4
   ```
2. Increase training epochs
3. Check training data quality
4. Verify data format is correct
5. Try different LoRA rank:
   ```python
   r=32  # Instead of 16
   ```

#### Problem: Training is very slow
**Symptoms:** Takes much longer than expected

**Solutions:**
1. Verify CUDA is being used:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
2. Increase batch size if VRAM allows
3. Reduce sequence length
4. Use mixed precision (already enabled with bf16)
5. Check GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

### Data Collection Issues

#### Problem: PubMed API errors
**Symptoms:** No papers downloaded from PubMed

**Solutions:**
1. Verify `PUBMED_EMAIL` is set in `.env`
2. Check email format is valid
3. Wait and retry (rate limiting)
4. Check internet connection
5. Verify BioPython is installed:
   ```bash
   pip install biopython
   ```

#### Problem: arXiv search returns no results
**Symptoms:** No arXiv papers found

**Solutions:**
1. Check internet connection
2. Verify arxiv package is installed:
   ```bash
   pip install arxiv
   ```
3. Try different search queries
4. Check arXiv API status

#### Problem: Too few papers downloaded
**Symptoms:** Less than 50 papers total

**Solutions:**
1. Increase `max_results` in search functions
2. Add more search queries
3. Expand date ranges in PubMed queries
4. Check for API rate limiting

---

### Evaluation Issues

#### Problem: Script 2 fails with import error
**Symptoms:** Cannot import questions

**Solutions:**
1. Ensure all scripts are in `scripts/` directory
2. Run from project root directory
3. Check Python path includes project root

#### Problem: Results file not found
**Symptoms:** Script 8 can't find results files

**Solutions:**
1. Ensure previous scripts completed successfully
2. Check `RESULTS_DIR` in `.env`
3. Verify files exist:
   ```bash
   ls results/
   ```
4. Run missing scripts first

---

### Performance Issues

#### Problem: Inference is slow
**Symptoms:** Takes >10 seconds per question

**Solutions:**
1. Reduce `max_new_tokens` in generation
2. Use 4-bit quantization (already enabled)
3. Enable model caching
4. Use GPU instead of CPU
5. Consider merging LoRA adapters for faster inference

#### Problem: Model responses are poor quality
**Symptoms:** Irrelevant or incorrect answers

**Solutions:**
1. Increase training epochs
2. Add more training data
3. Adjust learning rate
4. Try full fine-tuning instead of QLoRA
5. Check training data quality
6. Verify prompt format matches training format

---

### File System Issues

#### Problem: Permission denied errors
**Symptoms:** Cannot write to directories

**Solutions:**
1. Check directory permissions:
   ```bash
   ls -la
   ```
2. Create directories manually:
   ```bash
   mkdir -p data results models checkpoints logs
   ```
3. Fix permissions:
   ```bash
   chmod -R 755 .
   ```

#### Problem: Disk space full
**Symptoms:** Write errors or out of space

**Solutions:**
1. Check disk space:
   ```bash
   df -h
   ```
2. Clean old checkpoints:
   ```bash
   rm -rf checkpoints/qlora_checkpoint/checkpoint-*
   ```
3. Remove old model files if not needed
4. Use external storage for models

---

## Getting Help

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Status

```bash
# GPU status
nvidia-smi

# Conda version and environment
conda --version
conda info --envs

# Python version
python --version

# CUDA version
python -c "import torch; print(torch.version.cuda)"

# Installed packages
pip list | grep -E "(torch|transformers|peft)"

# Verify conda environment is active
echo $CONDA_DEFAULT_ENV
```

### Common Error Messages

#### "CUDA out of memory"
- Reduce batch size or sequence length
- Close other GPU processes
- Use gradient checkpointing

#### "ModuleNotFoundError"
- Activate conda environment: `conda activate Plan`
- Install missing package: `pip install package_name`

#### "FileNotFoundError"
- Check file paths in `.env`
- Ensure previous scripts completed
- Verify directory structure

#### "ConnectionError" or "TimeoutError"
- Check internet connection
- Retry after waiting
- Use VPN if in restricted region

---

## Best Practices for Debugging

1. **Run scripts individually**: Don't use `run_all.sh` when debugging
2. **Check logs**: Review `logs/` directory for detailed errors
3. **Verify inputs**: Ensure data files exist and are valid JSON
4. **Test incrementally**: Test each component separately
5. **Monitor resources**: Use `nvidia-smi` and `htop` to monitor usage
6. **Save checkpoints**: Enable checkpoint saving for recovery
7. **Version control**: Track changes to identify what broke

---

## Recovery Procedures

### After Training Crash

1. Check if checkpoint exists:
   ```bash
   ls checkpoints/qlora_checkpoint/
   ```
2. Resume from checkpoint (modify script to load checkpoint)
3. Or restart training (will overwrite)

### After Data Corruption

1. Re-download papers:
   ```bash
   python scripts/3_download_papers.py
   ```
2. Verify JSON files are valid:
   ```python
   import json
   with open('data/raw_papers.json') as f:
       json.load(f)  # Will raise error if invalid
   ```

### After Model Corruption

1. Re-download base model (will use cache if exists)
2. Re-run training script
3. Check disk space and permissions

