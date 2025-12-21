# Autoimmune Disease Diagnosis LLM Fine-Tuning

Complete pipeline for fine-tuning nvidia/Nemotron-Cascade-8B-Thinking on autoimmune disease research papers and evaluating performance on clinical benchmark questions.

## Features

- ✅ Automated model download and testing
- ✅ Benchmark question set for 10 autoimmune conditions
- ✅ Automated research paper collection (PubMed + arXiv)
- ✅ QLoRA fine-tuning (memory-efficient)
- ✅ Optional full fine-tuning
- ✅ Comprehensive evaluation metrics
- ✅ Automated comparison and reporting

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.8 support (for training)
- Python 3.10+ (recommended: 3.11+)

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
# (Optional, recommended) use venv:
# USE_VENV=1 ./setup.sh
# source .venv/bin/activate
```

The setup script will:
- Install dependencies into your current Python environment by default (Conda `(base)` is fine)
- Optionally create a Python virtual environment at `.venv` (set `USE_VENV=1`)
- Install PyTorch 2.7.1 with CUDA 12.8 support via pip
- Install all required dependencies via pip
- Create necessary directories

### 2. Configure

Edit `.env` and add your email for PubMed API:
```bash
PUBMED_EMAIL=your.email@example.com
```

### 3. Run Pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

**Note:** The scripts run in your current environment by default. If you want to force using `.venv`, run with `USE_VENV=1` (and make sure `.venv` exists).

## Hardware Requirements

### Minimum (QLoRA only)

- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 50GB free

### Recommended (Full fine-tuning)

- GPU: 3x NVIDIA RTX 3090 (72GB total VRAM)
- RAM: 64GB
- Storage: 100GB free

## Pipeline Steps

1. **Download & Test** (~15 min) - Download model, run inference test
2. **Benchmark Testing** (~20 min) - Test on 10 autoimmune questions
3. **Download Papers** (~10 min) - Fetch 100+ research papers
4. **QLoRA Fine-tuning** (~1-2 hours) - Memory-efficient training
5. **Test QLoRA** (~20 min) - Evaluate QLoRA model
6. **Full Fine-tuning** (~4-8 hours, optional) - Train all parameters
7. **Test Full Model** (~20 min) - Evaluate full model
8. **Score & Compare** (~5 min) - Generate comparison report

## Expected Results

### Output Files

- `baseline_results.json` - Pre-training performance
- `qlora_results.json` - QLoRA performance
- `full_model_results.json` - Full fine-tuning performance
- `model_comparison.csv` - Metric comparison table
- `final_report.txt` - Executive summary

### Performance Metrics

- Response time (seconds)
- Word count
- Reasoning quality (presence of <think> tags)
- Medical terminology usage
- Structured output format
- Confidence reporting
- Next steps suggestions

## Customization

### Add Custom Questions

Edit `scripts/2_autoimmune_questions.py` and modify `AUTOIMMUNE_QUESTIONS` list.

### Adjust Training Parameters

Edit `.env` file to modify:

- Learning rates
- Batch sizes
- Number of epochs
- LoRA configuration

### Change Base Model

Modify `MODEL_NAME` in `.env` to use different model.

## Troubleshooting

### Out of Memory Errors

- Reduce `QLORA_BATCH_SIZE` in `.env` (try `1`)
- Increase `QLORA_GRAD_ACCUM` in `.env` to keep effective batch size similar (e.g. `16`)
- Reduce `MAX_SEQ_LENGTH` in `.env` (try `1024` if still OOM)
- Use 4-bit quantization for inference

### Slow Training

- Enable gradient checkpointing
- Use mixed precision (bf16)
- Reduce sequence length

### Poor Results

- Increase training epochs
- Add more training data
- Adjust learning rate
- Try full fine-tuning instead of QLoRA

## Citation

```bibtex
@software{autoimmune_llm_2025,
  title = {Autoimmune Disease Diagnosis LLM Fine-Tuning Pipeline},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/autoimmune-llm}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- DeepSeek-AI for DeepSeek-R1 model
- Hugging Face for Transformers library
- Meta for Llama and PEFT libraries
