# Nemotron-Cascade-8B-Thinking Setup Guide

This document explains the changes made to support the `nvidia/Nemotron-Cascade-8B-Thinking` model and how to use it.

## Changes Made

### 1. Model Configuration
- **Default Model**: Changed from `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` to `nvidia/Nemotron-Cascade-8B-Thinking`
- Updated in `scripts/common.py` (Config.MODEL_NAME)
- Updated in `setup.sh` (.env template)

### 2. Prompt Formatting
The `format_prompt()` function in `scripts/common.py` has been updated to support multiple model formats:

- **Nemotron-Cascade**: Uses chat template with `/think` or `/no_think` control tokens
  - `/think`: Deep thinking mode (generates detailed reasoning steps)
  - `/no_think`: Instruct mode (provides direct answers)
- **DeepSeek**: Legacy format still supported for backward compatibility
- **Other models**: Automatically uses tokenizer's chat template if available

### 3. Updated Scripts
All scripts that use `format_prompt()` have been updated to pass the tokenizer:
- `scripts/1_download_and_test.py`
- `scripts/2_autoimmune_questions.py`
- `scripts/5_test_qlora.py`
- `scripts/7_test_full_model.py`

## Usage

### Basic Testing

1. **Download and test the model**:
   ```bash
   python scripts/1_download_and_test.py
   ```

2. **Run baseline benchmark** (tests with autoimmune questions):
   ```bash
   python scripts/2_autoimmune_questions.py
   ```

### Fine-Tuning

The fine-tuning scripts (`4_finetune.py` and `4_qlora_finetune.py`) work with Nemotron using:
- **DoRA** (Weight-Decomposed Low-Rank Adaptation) - recommended
- **QLoRA** (4-bit quantization + LoRA) - alternative

Both methods use standard attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) which are compatible with Nemotron's architecture.

### Thinking Mode

Nemotron-Cascade-8B-Thinking supports two modes:

1. **Thinking Mode** (`/think`): Default mode that generates detailed reasoning steps before the final answer. Useful for complex medical questions.

2. **Direct Mode** (`/no_think`): Provides direct answers without detailed reasoning. Faster but less transparent.

The default is thinking mode. To change this, modify the `use_thinking` parameter in `format_prompt()` calls.

## Configuration

### Environment Variables

Create or edit `.env` file:

```bash
# Model Configuration
MODEL_NAME=nvidia/Nemotron-Cascade-8B-Thinking
MAX_SEQ_LENGTH=1024  # recommended for ~12GB VRAM laptops; raise to 2048 if it fits

# Training Configuration
QLORA_EPOCHS=3
QLORA_BATCH_SIZE=2
QLORA_GRAD_ACCUM=8
QLORA_LR=2e-4
```

### Customizing Thinking Mode

To use direct mode instead of thinking mode, you can modify the `format_prompt()` calls:

```python
# In scripts, change:
prompt = format_prompt(question, tokenizer=tokenizer)
# To:
prompt = format_prompt(question, tokenizer=tokenizer, use_thinking=False)
```

## Pipeline Workflow

1. **Download & Test** (`1_download_and_test.py`)
   - Downloads Nemotron-Cascade-8B-Thinking
   - Runs basic inference test

2. **Baseline Benchmark** (`2_autoimmune_questions.py`)
   - Tests model on autoimmune questions
   - Creates `baseline_results.json`

3. **Download Papers** (`3_download_papers.py`)
   - Fetches research papers from PubMed and arXiv
   - Creates training dataset

4. **Fine-Tune** (`4_finetune.py` or `4_qlora_finetune.py`)
   - Trains model on research papers
   - Uses DoRA or QLoRA for memory efficiency

5. **Test Fine-Tuned Model** (`5_test_qlora.py`)
   - Evaluates fine-tuned model on same questions
   - Creates `finetuned_results.json`

6. **Compare Results** (`8_score_results.py`)
   - Compares baseline vs fine-tuned performance
   - Generates comparison report

## Expected Improvements

After fine-tuning on autoimmune research papers, you should see improvements in:
- Medical terminology accuracy
- Clinical reasoning quality
- Response relevance to autoimmune diseases
- Structured output format

## Troubleshooting

### Model Loading Issues
- Ensure you have sufficient VRAM (24GB+ recommended for 4-bit quantization)
- Check Hugging Face authentication if model is gated

### Prompt Format Issues
- The code automatically detects Nemotron models and uses appropriate format
- If issues occur, check that tokenizer has `apply_chat_template` method

### Training Issues
- Nemotron uses standard transformer architecture, so LoRA/DoRA target modules should work
- If training fails, try reducing batch size or using gradient checkpointing

## Notes

- The model supports both thinking and direct modes via control tokens
- Thinking mode generates more detailed responses but is slower
- Fine-tuning preserves the model's reasoning capabilities while adding domain knowledge

