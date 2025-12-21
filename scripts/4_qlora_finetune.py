#!/usr/bin/env python3
"""
Script 4: QLoRA Fine-Tuning
Version: 1.1.0

Purpose: Fine-tune model using QLoRA (memory-efficient LoRA with 4-bit quantization).

Usage:
    python scripts/4_qlora_finetune.py

Input:
    - data/papers_training_data.json: Training data from script 3

Output:
    - models/qlora_model/: Trained LoRA adapters
    - checkpoints/qlora_checkpoint/: Training checkpoints
"""

import sys
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

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    load_json,
    get_quantization_config,
    print_header,
    print_section,
    print_cuda_info,
    get_vram_usage,
    confirm_continue,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(logger) -> list:
    """Load training data from JSON file."""
    data_file = Config.DATA_DIR / "papers_training_data.json"
    
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        print(f"✗ ERROR: Training data file not found: {data_file}")
        print("Please run script 3 first to download papers.")
        sys.exit(1)
    
    data = load_json(data_file)
    texts = [item['text'] for item in data]
    
    print(f"✓ Loaded {len(texts)} training examples")
    logger.info(f"Loaded {len(texts)} training examples")
    
    return texts


# =============================================================================
# MODEL SETUP
# =============================================================================

def load_model_for_training(logger):
    """Load model and tokenizer for QLoRA training."""
    print(f"\nLoading model: {Config.MODEL_NAME}...")
    logger.info(f"Loading model: {Config.MODEL_NAME}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=get_quantization_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Extra safety: cache can blow up memory during training
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    
    vram = get_vram_usage()
    print(f"✓ Model loaded, VRAM: {vram['allocated']:.2f} GB")
    logger.info(f"Model loaded, VRAM: {vram['allocated']:.2f} GB")
    
    return model, tokenizer


def create_lora_config() -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(texts: list, tokenizer, logger) -> Dataset:
    """Prepare dataset for training."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            padding=False
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    # Add a length column to enable efficient bucketing (reduces padding/VRAM)
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"length": len(x["input_ids"])},
        desc="Computing lengths"
    )
    
    logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
    return tokenized_dataset


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, tokenizer, dataset, logger):
    """Train the model with QLoRA."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Output directory
    output_dir = Config.CHECKPOINTS_DIR / "qlora_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=Config.QLORA_EPOCHS,
        per_device_train_batch_size=Config.QLORA_BATCH_SIZE,
        gradient_accumulation_steps=Config.QLORA_GRAD_ACCUM,
        learning_rate=Config.QLORA_LR,
        bf16=True,
        logging_steps=10,
        logging_dir=str(Config.LOGS_DIR / "tensorboard"),
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Print training info
    print_section("STARTING QLORA TRAINING")
    effective_batch = Config.QLORA_BATCH_SIZE * Config.QLORA_GRAD_ACCUM
    print(f"Epochs: {Config.QLORA_EPOCHS}")
    print(f"Batch Size: {Config.QLORA_BATCH_SIZE}")
    print(f"Gradient Accumulation: {Config.QLORA_GRAD_ACCUM}")
    print(f"Effective Batch Size: {effective_batch}")
    print(f"Learning Rate: {Config.QLORA_LR}")
    print(f"Training Examples: {len(dataset)}")
    print(f"Total Steps: {len(dataset) // effective_batch * Config.QLORA_EPOCHS}")
    
    logger.info(f"Starting training: {Config.QLORA_EPOCHS} epochs, {len(dataset)} examples")
    
    # Train
    trainer.train()
    
    logger.info("Training completed")
    return trainer


def save_model(model, tokenizer, logger):
    """Save the trained LoRA model."""
    output_path = Config.MODELS_DIR / "qlora_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Model saved to {output_path}")
    logger.info(f"Model saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print_header("SCRIPT 4: QLORA FINE-TUNING")
    
    # Setup
    logger = setup_logging("4_qlora_finetune")
    Config.ensure_directories()
    
    # Check CUDA
    cuda_info = print_cuda_info(logger)
    
    if not cuda_info["available"]:
        logger.warning("CUDA not available")
        print("\n⚠ WARNING: CUDA is not available. Training will be extremely slow.")
        if not confirm_continue("Continue anyway?"):
            sys.exit(0)
    
    try:
        # Load training data
        texts = load_training_data(logger)
        
        if len(texts) < 10:
            print("\n⚠ WARNING: Very few training examples. Results may be poor.")
            logger.warning(f"Only {len(texts)} training examples")
        
        # Load model
        model, tokenizer = load_model_for_training(logger)
        
        # Apply LoRA
        print("\nApplying LoRA adapters...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")
        
        # Prepare dataset
        dataset = prepare_dataset(texts, tokenizer, logger)
        
        # Train
        trainer = train_model(model, tokenizer, dataset, logger)
        
        # Save model
        save_model(model, tokenizer, logger)
        
        # Print final VRAM usage
        vram = get_vram_usage()
        print(f"\nFinal VRAM Usage: {vram['allocated']:.2f} GB")
        
        print_header("✓ Script 4 completed successfully!")
        logger.info("Script completed successfully")
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA Out of Memory")
        print("\n✗ CUDA Out of Memory!")
        print("Try one or more of the following in your .env:")
        print("  - QLORA_BATCH_SIZE=1")
        print("  - Increase QLORA_GRAD_ACCUM (e.g. 16 or 32) to keep effective batch similar")
        print("  - Reduce MAX_SEQ_LENGTH (e.g. 1024)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
