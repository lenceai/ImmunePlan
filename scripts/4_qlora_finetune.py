#!/usr/bin/env python3
"""
Script 4: QLoRA Fine-Tuning
Version: 1.0.0

Purpose: Fine-tune model using QLoRA (memory-efficient LoRA with 4-bit quantization).

Usage:
    python scripts/4_qlora_finetune.py

Input:
    - data/papers_training_data.json: Training data from script 3

Output:
    - models/qlora_model/: Trained LoRA adapters
    - checkpoints/qlora_checkpoint/: Training checkpoints
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
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
QLORA_EPOCHS = int(os.getenv("QLORA_EPOCHS", "3"))
QLORA_BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "2"))
QLORA_LR = float(os.getenv("QLORA_LR", "2e-4"))

# Ensure directories exist
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path("checkpoints/qlora_checkpoint").mkdir(parents=True, exist_ok=True)


def load_training_data():
    """
    Load training data from JSON file.
    
    Returns:
        List of training texts
    """
    data_file = os.path.join(DATA_DIR, "papers_training_data.json")
    
    if not os.path.exists(data_file):
        print(f"ERROR: Training data file not found: {data_file}")
        print("Please run script 3 first to download papers.")
        sys.exit(1)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract text fields
    texts = [item['text'] for item in data]
    
    print(f"Loaded {len(texts)} training examples")
    return texts


def load_model_and_tokenizer():
    """
    Load model and tokenizer with 4-bit quantization.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading model: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        max_length=MAX_SEQ_LENGTH
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def create_lora_config():
    """
    Create LoRA configuration.
    
    Returns:
        LoraConfig object
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    return lora_config


def prepare_dataset(texts, tokenizer):
    """
    Prepare dataset for training.
    
    Args:
        texts: List of training texts
        tokenizer: Tokenizer
    
    Returns:
        Dataset object
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """
    Train the model with QLoRA.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        dataset: Training dataset
    """
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/qlora_checkpoint",
        num_train_epochs=QLORA_EPOCHS,
        per_device_train_batch_size=QLORA_BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=QLORA_LR,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb/tensorboard by default
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("STARTING QLORA TRAINING")
    print(f"{'='*60}\n")
    print(f"Epochs: {QLORA_EPOCHS}")
    print(f"Batch Size: {QLORA_BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: 8")
    print(f"Effective Batch Size: {QLORA_BATCH_SIZE * 8}")
    print(f"Learning Rate: {QLORA_LR}")
    print(f"Training Examples: {len(dataset)}\n")
    
    trainer.train()
    
    return trainer


def save_model(model, tokenizer, output_dir):
    """
    Save the trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    output_path = os.path.join(MODELS_DIR, "qlora_model")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Model saved to {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 4: QLORA FINE-TUNING")
    print("="*60 + "\n")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    try:
        # Load training data
        texts = load_training_data()
        
        if len(texts) < 10:
            print("WARNING: Very few training examples. Results may be poor.")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Create LoRA config
        lora_config = create_lora_config()
        
        # Apply LoRA
        print("\nApplying LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Prepare dataset
        print("\nPreparing dataset...")
        dataset = prepare_dataset(texts, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, dataset)
        
        # Save model
        save_model(model, tokenizer, MODELS_DIR)
        
        print("\n" + "="*60)
        print("✓ Script 4 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

