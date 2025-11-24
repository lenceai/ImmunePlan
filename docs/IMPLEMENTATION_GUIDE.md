# Implementation Guide

## Overview

This guide provides detailed information about implementing and customizing the Autoimmune LLM Fine-Tuning Pipeline.

## Architecture

### Pipeline Flow

```
1. Download & Test → 2. Baseline Test → 3. Download Papers
                                              ↓
8. Score Results ← 7. Test Full Model ← 6. Full Fine-Tune
                                              ↑
                                   5. Test QLoRA ← 4. QLoRA Fine-Tune
```

### Key Components

1. **Model Loading**: Uses 4-bit quantization for memory efficiency
2. **Data Collection**: Automated PubMed and arXiv paper fetching
3. **Training**: QLoRA for efficient fine-tuning, optional full fine-tuning
4. **Evaluation**: Comprehensive benchmark on 10 autoimmune questions
5. **Comparison**: Automated metrics and reporting

## Customization

### Adding Custom Questions

Edit `scripts/2_autoimmune_questions.py`:

```python
AUTOIMMUNE_QUESTIONS.append({
    "id": "Q011",
    "category": "Your Category",
    "difficulty": "medium",
    "question": "Your question here..."
})
```

### Modifying Training Parameters

Edit `.env` file:

```bash
# QLoRA Training
QLORA_EPOCHS=5              # Increase epochs
QLORA_BATCH_SIZE=4          # Increase batch size (if VRAM allows)
QLORA_LR=1e-4               # Adjust learning rate

# Full Fine-Tuning
FULL_EPOCHS=5
FULL_BATCH_SIZE=2
FULL_LR=1e-5
```

### Changing Base Model

Edit `.env`:

```bash
MODEL_NAME=your-model-name
```

Ensure the model supports:
- 4-bit quantization (for QLoRA)
- Causal LM task type
- Similar architecture to DeepSeek-R1

### Adjusting LoRA Configuration

Edit `scripts/4_qlora_finetune.py`:

```python
lora_config = LoraConfig(
    r=32,                    # Increase rank
    lora_alpha=64,           # Increase alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Add more modules
    lora_dropout=0.1,        # Increase dropout
    ...
)
```

## Data Format

### Training Data Format

The training data JSON should have this structure:

```json
[
    {
        "text": "User: Question here...RetryRMfinish",
        "source": "PubMed",
        "paper_id": "12345678",
        "title": "Paper Title"
    }
]
```

### Results Format

Results are stored as:

```json
[
    {
        "id": "Q001",
        "category": "Systemic Lupus Erythematosus",
        "difficulty": "medium",
        "question": "...",
        "response": "...",
        "time_seconds": 2.34,
        "timestamp": "2025-01-01T12:00:00"
    }
]
```

## Performance Optimization

### Memory Optimization

1. **Reduce Batch Size**: Lower `QLORA_BATCH_SIZE` or `FULL_BATCH_SIZE`
2. **Increase Gradient Accumulation**: Compensate for smaller batches
3. **Use Gradient Checkpointing**: Already enabled by default
4. **Reduce Sequence Length**: Lower `MAX_SEQ_LENGTH` in `.env`

### Speed Optimization

1. **Use Mixed Precision**: Already using bf16
2. **Enable Flash Attention**: If supported by model
3. **Reduce Epochs**: For faster iteration
4. **Use QLoRA Instead of Full Fine-Tuning**: Much faster

## Integration

### Using Trained Models

#### QLoRA Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./models/qlora_model")
tokenizer = AutoTokenizer.from_pretrained("./models/qlora_model")
```

#### Full Model

```python
model = AutoModelForCausalLM.from_pretrained("./models/full_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./models/full_finetuned_model")
```

### API Integration

Create a simple API wrapper:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load model here...

@app.route('/predict', methods=['POST'])
def predict():
    question = request.json['question']
    response = get_model_response(model, tokenizer, question)
    return jsonify({'response': response})
```

## Best Practices

1. **Always test baseline first**: Understand model behavior before fine-tuning
2. **Start with QLoRA**: More efficient and often sufficient
3. **Monitor training**: Check logs for loss and convergence
4. **Validate on held-out data**: Don't overfit to training questions
5. **Save checkpoints**: Enable checkpoint saving for recovery
6. **Version control**: Track model versions and hyperparameters

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.

