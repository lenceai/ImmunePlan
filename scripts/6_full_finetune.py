#!/usr/bin/env python3
"""
Script 6: Full Fine-Tuning
Version: 1.0.0

Purpose: Full fine-tuning of the model (all parameters). Requires significant VRAM.

Usage:
    python scripts/6_full_finetune.py

Warning: This script requires 3x RTX 3090 (72GB VRAM) or equivalent.
         It is optional and can be skipped for MVP.

Input:
    - data/papers_training_data.json: Training data

Output:
    - models/full_finetuned_model/: Fully fine-tuned model
    - checkpoints/full_finetune_checkpoint/: Training checkpoints
"""

import os
import sys
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
FULL_EPOCHS = int(os.getenv("FULL_EPOCHS", "3"))
FULL_BATCH_SIZE = int(os.getenv("FULL_BATCH_SIZE", "1"))
FULL_LR = float(os.getenv("FULL_LR", "5e-5"))

# Ensure directories exist
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path("checkpoints/full_finetune_checkpoint").mkdir(parents=True, exist_ok=True)


def check_vram():
    """Check if sufficient VRAM is available."""
    if not torch.cuda.is_available():
        return False, 0
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Require at least 40GB for full fine-tuning
    if total_vram < 40:
        return False, total_vram
    
    return True, total_vram


def load_training_data():
    """Load training data from JSON file."""
    data_file = os.path.join(DATA_DIR, "papers_training_data.json")
    
    if not os.path.exists(data_file):
        print(f"ERROR: Training data file not found: {data_file}")
        sys.exit(1)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    print(f"Loaded {len(texts)} training examples")
    return texts


def load_model_and_tokenizer():
    """Load model and tokenizer for full fine-tuning."""
    print(f"\nLoading model: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model without quantization for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_length=MAX_SEQ_LENGTH
    )
    
    return model, tokenizer


def prepare_dataset(texts, tokenizer):
    """Prepare dataset for training."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model with full fine-tuning."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir="checkpoints/full_finetune_checkpoint",
        num_train_epochs=FULL_EPOCHS,
        per_device_train_batch_size=FULL_BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=FULL_LR,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print(f"\n{'='*60}")
    print("STARTING FULL FINE-TUNING")
    print(f"{'='*60}\n")
    print(f"Epochs: {FULL_EPOCHS}")
    print(f"Batch Size: {FULL_BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: 16")
    print(f"Effective Batch Size: {FULL_BATCH_SIZE * 16}")
    print(f"Learning Rate: {FULL_LR}")
    print(f"Training Examples: {len(dataset)}\n")
    
    trainer.train()
    
    return trainer


def save_model(model, tokenizer, output_dir):
    """Save the fully fine-tuned model."""
    output_path = os.path.join(MODELS_DIR, "full_finetuned_model")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Model saved to {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 6: FULL FINE-TUNING")
    print("="*60 + "\n")
    
    # Check VRAM
    has_vram, total_vram = check_vram()
    
    if not has_vram:
        print(f"WARNING: Insufficient VRAM detected ({total_vram:.2f} GB)")
        print("Full fine-tuning requires at least 40GB VRAM (recommended: 72GB)")
        print("\nThis script is OPTIONAL. You can skip it and use QLoRA results.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Skipping full fine-tuning. Use QLoRA model for evaluation.")
            sys.exit(0)
    
    print(f"✓ VRAM Available: {total_vram:.2f} GB")
    
    try:
        # Load training data
        texts = load_training_data()
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable Parameters: {trainable_params:,}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Prepare dataset
        print("\nPreparing dataset...")
        dataset = prepare_dataset(texts, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, dataset)
        
        # Save model
        save_model(model, tokenizer, MODELS_DIR)
        
        print("\n" + "="*60)
        print("✓ Script 6 completed successfully!")
        print("="*60 + "\n")
        
    except torch.cuda.OutOfMemoryError:
        print("\n✗ Out of Memory Error!")
        print("Full fine-tuning requires more VRAM.")
        print("Consider using QLoRA instead (script 4).")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

