#!/usr/bin/env python3
"""
Script 6: High-Rank LoRA Fine-Tuning (Single 24GB GPU Optimized)
Version: 1.7.0

Purpose: Fine-tuning using 4-bit quantization with high-rank LoRA adapters.
         Uses much higher LoRA rank (256) and targets all linear layers for
         maximum training capacity while fitting in 24GB VRAM.
         
         This provides significantly more trainable parameters than standard
         QLoRA (rank 64) while still benefiting from 4-bit quantization.

Usage:
    python scripts/6_full_finetune.py

Requirements: Single GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)

Input:
    - data/papers_training_data.json: Training data

Output:
    - models/full_finetuned_model/: Fine-tuned model with merged weights
    - checkpoints/full_finetune_checkpoint/: Training checkpoints
"""

import sys
import gc
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    load_json,
    check_cuda,
    print_header,
    print_section,
    print_cuda_info,
    get_vram_usage,
    clear_gpu_memory,
    confirm_continue,
)

# Memory optimization settings - optimized for 24GB with 4-bit + high-rank LoRA
FULL_FINETUNE_SEQ_LENGTH = 512  # Balanced for rank 384 LoRA
GRADIENT_ACCUMULATION_STEPS = 32  # Good accumulation
FULL_FINETUNE_BATCH_SIZE = 1  # Keep at 1 for memory

# Very high-rank LoRA settings - maximum capacity for 24GB GPU
LORA_RANK = 384  # High rank (standard QLoRA uses 64) - ~970M trainable params
LORA_ALPHA = 768  # Alpha = 2 * rank is common
LORA_DROPOUT = 0.05


# =============================================================================
# VRAM CHECK
# =============================================================================

def check_vram_requirements(logger) -> tuple:
    """Check if sufficient VRAM is available for single-GPU full fine-tuning."""
    cuda_info = check_cuda()
    
    if not cuda_info["available"]:
        return False, 0, "CUDA not available"
    
    total_vram = cuda_info["total_vram_gb"]
    
    # Require at least 20GB for optimized full fine-tuning of 7B model
    if total_vram < 20:
        return False, total_vram, f"Insufficient VRAM ({total_vram:.1f}GB < 20GB required)"
    
    return True, total_vram, "OK"


# =============================================================================
# DATA & MODEL
# =============================================================================

def load_training_data(logger) -> list:
    """Load training data from JSON file."""
    data_file = Config.DATA_DIR / "papers_training_data.json"
    
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        print(f"✗ ERROR: Training data file not found: {data_file}")
        sys.exit(1)
    
    data = load_json(data_file)
    texts = [item['text'] for item in data]
    
    print(f"✓ Loaded {len(texts)} training examples")
    logger.info(f"Loaded {len(texts)} training examples")
    
    return texts


def get_lora_config():
    """
    Get high-rank LoRA configuration for maximum training capacity.
    
    This configuration targets ALL linear layers and uses a high rank
    to maximize trainable parameters while still fitting in 24GB.
    """
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Target ALL linear layers for maximum capacity
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )


def get_4bit_config():
    """Get 4-bit quantization configuration for training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_for_full_training(logger):
    """
    Load model with 4-bit quantization and high-rank LoRA adapters.
    
    This approach:
    - Uses 4-bit NF4 quantization (reduces model from ~14GB to ~4GB)
    - Adds high-rank LoRA adapters (rank 256) to all linear layers
    - Results in ~1.3B trainable parameters (vs ~40M for standard QLoRA)
    """
    print(f"\nLoading model: {Config.MODEL_NAME}...")
    print("Using 4-bit quantization + high-rank LoRA...")
    logger.info(f"Loading model with 4-bit + high-rank LoRA: {Config.MODEL_NAME}")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    gc.collect()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config
    quant_config = get_4bit_config()
    
    # Model WITH 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    
    vram = get_vram_usage()
    print(f"✓ Base model loaded (4-bit), VRAM: {vram['allocated']:.2f} GB / {vram['total']:.2f} GB")
    logger.info(f"Base model loaded with 4-bit quantization, VRAM: {vram['allocated']:.2f} GB")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    print("✓ Model prepared for k-bit training")
    
    # Add high-rank LoRA adapters
    print(f"\nAdding high-rank LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    vram = get_vram_usage()
    print(f"✓ LoRA adapters added, VRAM: {vram['allocated']:.2f} GB / {vram['total']:.2f} GB")
    logger.info(f"LoRA adapters added, VRAM: {vram['allocated']:.2f} GB")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Ensure model is in training mode
    model.train()
    
    return model, tokenizer


def prepare_dataset(texts: list, tokenizer, logger) -> Dataset:
    """
    Prepare dataset for training with memory-optimized settings.
    
    Uses reduced sequence length to fit in 24GB VRAM.
    """
    print(f"Tokenizing dataset with max_length={FULL_FINETUNE_SEQ_LENGTH}...")
    print(f"(Reduced from {Config.MAX_SEQ_LENGTH} for memory efficiency)")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=FULL_FINETUNE_SEQ_LENGTH,  # Reduced sequence length
            padding="max_length"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples, max_length={FULL_FINETUNE_SEQ_LENGTH}")
    return tokenized_dataset


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, tokenizer, dataset, logger):
    """
    Train the model with memory-optimized full fine-tuning.
    
    Key optimizations for 24GB GPU:
    - 8-bit Adam optimizer (adamw_bnb_8bit) - reduces optimizer memory by ~50%
    - Aggressive gradient accumulation (64 steps)
    - Gradient checkpointing (already enabled on model)
    - bf16 mixed precision
    - Reduced batch size (1)
    - Reduced sequence length (256)
    """
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    output_dir = Config.CHECKPOINTS_DIR / "full_finetune_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate effective batch size
    effective_batch = FULL_FINETUNE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=Config.FULL_EPOCHS,
        per_device_train_batch_size=FULL_FINETUNE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.FULL_LR,
        
        # Memory optimizations
        bf16=True,
        optim="adamw_bnb_8bit",  # 8-bit Adam - crucial for memory savings
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Reduce memory fragmentation
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
        # Logging and saving
        logging_steps=5,
        logging_dir=str(Config.LOGS_DIR / "tensorboard_full"),
        save_strategy="epoch",
        save_total_limit=2,
        
        # Learning rate schedule
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Disable reporting to save memory
        report_to="none",
        
        # Additional optimizations
        eval_strategy="no",
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Print training info
    print_section("STARTING HIGH-RANK LORA FINE-TUNING")
    print(f"Target: Single 24GB GPU")
    print(f"Quantization: 4-bit NF4 with double quantization")
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Optimizer: 8-bit AdamW (memory optimized)")
    print(f"Epochs: {Config.FULL_EPOCHS}")
    print(f"Batch Size: {FULL_FINETUNE_BATCH_SIZE}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch Size: {effective_batch}")
    print(f"Sequence Length: {FULL_FINETUNE_SEQ_LENGTH}")
    print(f"Learning Rate: {Config.FULL_LR}")
    print(f"Training Examples: {len(dataset)}")
    
    # Show current VRAM before training
    vram = get_vram_usage()
    print(f"\nVRAM before training: {vram['allocated']:.2f} GB / {vram['total']:.2f} GB")
    
    logger.info(f"Starting high-rank LoRA fine-tuning: {Config.FULL_EPOCHS} epochs, rank={LORA_RANK}")
    logger.info(f"Effective batch size: {effective_batch}, seq_length: {FULL_FINETUNE_SEQ_LENGTH}")
    
    # Clear cache before training
    clear_gpu_memory()
    
    trainer.train()
    
    logger.info("Training completed")
    return trainer


def save_model(model, tokenizer, logger):
    """Save the fine-tuned model (LoRA adapters + optionally merged)."""
    # Save LoRA adapters
    adapter_path = Config.MODELS_DIR / "full_finetuned_lora"
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    print(f"\n✓ LoRA adapters saved to {adapter_path}")
    logger.info(f"LoRA adapters saved to {adapter_path}")
    
    # Merge and save full model
    print("\nMerging LoRA weights into base model...")
    try:
        merged_model = model.merge_and_unload()
        
        merged_path = Config.MODELS_DIR / "full_finetuned_model"
        merged_path.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        
        print(f"✓ Merged model saved to {merged_path}")
        logger.info(f"Merged model saved to {merged_path}")
    except Exception as e:
        print(f"⚠ Could not merge model: {e}")
        print("  LoRA adapters are still saved and can be loaded separately.")
        logger.warning(f"Could not merge model: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print_header("SCRIPT 6: HIGH-RANK LORA FINE-TUNING (Single 24GB GPU)")
    
    # Setup
    logger = setup_logging("6_full_finetune")
    Config.ensure_directories()
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check VRAM requirements
    print_cuda_info(logger)
    
    has_vram, total_vram, message = check_vram_requirements(logger)
    
    if not has_vram:
        print(f"\n⚠ WARNING: {message}")
        print("High-rank LoRA fine-tuning requires at least 20GB VRAM")
        print("\nThis script is OPTIONAL. You can skip it and use QLoRA results.")
        logger.warning(message)
        
        if not confirm_continue("Continue anyway?"):
            print("Skipping fine-tuning. Use QLoRA model for evaluation.")
            sys.exit(0)
    else:
        print(f"\n✓ VRAM Check: {total_vram:.1f} GB available (optimized for 24GB)")
    
    # Print optimization info
    print_section("HIGH-RANK LORA FINE-TUNING STRATEGY")
    print("• 4-bit NF4 quantization (model: ~14GB → ~4GB)")
    print(f"• High-rank LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    print("• Targets ALL linear layers (attention + MLP)")
    print("• ~30x more trainable params than standard QLoRA (rank 64)")
    print("• 8-bit AdamW optimizer (reduces optimizer memory ~50%)")
    print("• Gradient checkpointing (trades compute for memory)")
    print(f"• Sequence length: {FULL_FINETUNE_SEQ_LENGTH} tokens")
    print(f"• Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    
    try:
        # Load training data
        texts = load_training_data(logger)
        
        # Load model
        model, tokenizer = load_model_for_full_training(logger)
        
        # Print parameter info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable Parameters: {trainable_params:,}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        logger.info(f"Parameters: {trainable_params:,} / {total_params:,}")
        
        # Prepare dataset
        dataset = prepare_dataset(texts, tokenizer, logger)
        
        # Train
        trainer = train_model(model, tokenizer, dataset, logger)
        
        # Save model
        save_model(model, tokenizer, logger)
        
        print_header("✓ Script 6 completed successfully!")
        logger.info("Script completed successfully")
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory: {str(e)}")
        print("\n✗ Out of Memory Error!")
        print("\nTroubleshooting suggestions:")
        print(f"1. Reduce FULL_FINETUNE_SEQ_LENGTH (currently {FULL_FINETUNE_SEQ_LENGTH})")
        print(f"2. Increase GRADIENT_ACCUMULATION_STEPS (currently {GRADIENT_ACCUMULATION_STEPS})")
        print("3. Close other GPU applications")
        print("4. Use QLoRA instead (script 4) - more memory efficient")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
