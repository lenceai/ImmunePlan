#!/usr/bin/env python3
"""
Script 4: DoRA Fine-Tuning (Weight-Decomposed Low-Rank Adaptation)
Version: 2.0.0

Purpose: Fine-tune model using DoRA with 4-bit quantization.
         DoRA decomposes weights into magnitude and direction components,
         often achieving better performance than LoRA while using similar memory.

Usage:
    python scripts/4_finetune.py

Input:
    - data/papers_training_data.json: Training data from script 3

Output:
    - models/dora_model/: Trained DoRA adapters
    - checkpoints/dora_checkpoint/: Training checkpoints
"""

import os
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
    """Load model and tokenizer for DoRA training with 4-bit quantization."""
    print(f"\nLoading model: {Config.MODEL_NAME}...")
    print("Using 4-bit quantization + DoRA for maximum parameter efficiency")
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
    print(f"✓ Model loaded (4-bit), VRAM: {vram['allocated']:.2f} GB")
    logger.info(f"Model loaded, VRAM: {vram['allocated']:.2f} GB")
    
    return model, tokenizer


# ── Hardware tiers (rank + modules only — seq_len is handled separately) ──────
# Keyed on TOTAL GPU VRAM.  Thresholds are set 0.5 GB below the card's nominal
# size to tolerate firmware/driver reservations (e.g. 4080 Laptop = 11.6 GB).
# Each row: (min_total_vram_gb, rank, full_modules)
# full_modules True  → attention + MLP adapted
# full_modules False → attention only (~40 % less adapter activation memory)
_HW_TIERS = [
    (47.0, 64, True),    # dual 3090 / A100 80 GB
    (31.0, 32, True),    # single 3090 Ti / A6000
    (23.0, 32, True),    # single 3090 / 3080 Ti 24 GB
    (15.0, 16, True),    # 16 GB cards (3080 16 GB, 4080 Super)
    (11.0,  8, False),   # 11–15 GB — 4080 Laptop (11.6 GB), 3080 12 GB
    ( 7.0,  8, False),   # 8–11 GB cards
    ( 0.0,  4, False),   # < 8 GB / emergency
]

# Fixed seq_len ladder used for OOM retries — always tried in this order,
# starting from the first value that fits the hardware tier's starting point.
_SEQ_LADDER = [2048, 1024, 512, 384, 256, 128]

# Module-level state: index into _SEQ_LADDER for the current attempt
_seq_idx: int | None = None


def _total_vram_gb() -> float:
    """Sum of total VRAM across all GPUs in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        for i in range(torch.cuda.device_count())
    )


def _free_vram_gb() -> float:
    """Sum of unallocated VRAM across all GPUs in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        (torch.cuda.get_device_properties(i).total_memory
         - torch.cuda.memory_allocated(i)) / 1024 ** 3
        for i in range(torch.cuda.device_count())
    )


def auto_configure_vram(logger, reduce: bool = False) -> bool:
    """
    Select or reduce the training configuration and update Config + os.environ.

    First call (reduce=False):
        - Picks rank + modules from the hardware tier table (based on total VRAM).
        - Picks the most aggressive seq_len from _SEQ_LADDER that is ≤ the
          tier's natural starting point.

    Subsequent calls (reduce=True):
        - Steps down one position in _SEQ_LADDER (smaller seq_len).
        - rank + modules are never changed after the first call.

    Settings explicitly set in .env are never overridden.
    Returns False when the seq_len ladder is exhausted.
    """
    global _seq_idx

    explicit = {k: k in os.environ for k in ("DORA_RANK", "MAX_SEQ_LENGTH", "DORA_FULL_MODULES")}

    total_gb = _total_vram_gb()

    if not reduce:
        # ── Determine rank + modules from hardware tier ────────────────────────
        rank, full_modules = 4, False          # floor defaults
        for min_vram, r, f in _HW_TIERS:
            if total_gb >= min_vram:
                rank, full_modules = r, f
                break

        # ── Choose starting seq_len ────────────────────────────────────────────
        # Starting points per hardware tier (mirrors _HW_TIERS thresholds):
        #   ≥47 GB → 2048,  ≥23 GB → 1024,  ≥11 GB → 512,  else → 256
        if total_gb >= 47.0:
            start_seq = 2048
        elif total_gb >= 23.0:
            start_seq = 1024
        elif total_gb >= 11.0:
            start_seq = 512
        else:
            start_seq = 256

        # Find that seq_len in the ladder; fall back to the last entry if missing
        _seq_idx = len(_SEQ_LADDER) - 1
        for i, s in enumerate(_SEQ_LADDER):
            if s <= start_seq:
                _seq_idx = i
                break

        # Apply rank + modules (only if not pinned in .env)
        if not explicit["DORA_RANK"]:
            os.environ["DORA_RANK"] = str(rank)
        else:
            rank = int(os.environ["DORA_RANK"])

        if not explicit["DORA_FULL_MODULES"]:
            os.environ["DORA_FULL_MODULES"] = "1" if full_modules else "0"
        else:
            full_modules = os.environ["DORA_FULL_MODULES"] == "1"

    else:
        # ── OOM fallback: step down seq_len ladder ─────────────────────────────
        if _seq_idx is None or _seq_idx >= len(_SEQ_LADDER) - 1:
            return False                       # already at 128 — nothing left
        _seq_idx += 1
        rank = int(os.environ.get("DORA_RANK", "8"))
        full_modules = os.environ.get("DORA_FULL_MODULES", "0") == "1"

    seq_len = _SEQ_LADDER[_seq_idx]

    # Apply seq_len (only if not pinned in .env)
    if not explicit["MAX_SEQ_LENGTH"]:
        os.environ["MAX_SEQ_LENGTH"] = str(seq_len)
        Config.MAX_SEQ_LENGTH = seq_len
    else:
        seq_len = Config.MAX_SEQ_LENGTH

    # ── Print summary ──────────────────────────────────────────────────────────
    label = "VRAM AUTO-CONFIGURE" if not reduce else "OOM — stepping down seq_len"
    free_gb = _free_vram_gb()
    print("\n" + "─" * 60)
    print(f"  {label}")
    print("─" * 60)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
            tot   = props.total_memory / 1024 ** 3
            print(f"    GPU {i}: {props.name}  {alloc:.1f}/{tot:.1f} GB used")
        remaining = " → ".join(str(s) for s in _SEQ_LADDER[_seq_idx:])
        print(f"  Total VRAM: {total_gb:.1f} GB   Free: {free_gb:.1f} GB")
        print(f"  seq_len ladder remaining: {remaining}")
    print()
    for key, lbl in [
        ("DORA_RANK",         "DORA_RANK        "),
        ("MAX_SEQ_LENGTH",    "MAX_SEQ_LENGTH   "),
        ("DORA_FULL_MODULES", "DORA_FULL_MODULES"),
    ]:
        source = "(.env — pinned)" if explicit[key] else "(auto)"
        print(f"  {lbl} = {os.environ[key]:>6}  {source}")
    print("─" * 60)

    logger.info(
        f"VRAM config: rank={rank}, seq={seq_len}, full_modules={full_modules}, "
        f"free={free_gb:.1f} GB  [reduce={reduce}]"
    )
    return True


def create_dora_config() -> LoraConfig:
    """
    Create DoRA configuration using LoraConfig with use_dora=True.
    
    DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes weights into
    magnitude and direction components, often outperforming LoRA while
    using similar memory footprint.
    
    With 4-bit quantization, we can use higher rank to train more parameters.
    """
    rank = int(os.getenv("DORA_RANK", "8"))
    # On low-VRAM GPUs (<= 12 GB) target only attention projections.
    # Set DORA_FULL_MODULES=1 in .env to also adapt MLP layers (needs ~24 GB).
    full_modules = os.getenv("DORA_FULL_MODULES", "0") == "1"
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if full_modules:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=False,
        use_dora=True,
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
    """Train the model with DoRA."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Output directory
    output_dir = Config.CHECKPOINTS_DIR / "dora_checkpoint"
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
        optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
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
    print_section("STARTING DORA TRAINING")
    effective_batch = Config.QLORA_BATCH_SIZE * Config.QLORA_GRAD_ACCUM
    dora_rank = int(os.getenv("DORA_RANK", "8"))
    dora_full = os.getenv("DORA_FULL_MODULES", "0") == "1"
    print("Method:                DoRA (Weight-Decomposed Low-Rank Adaptation)")
    print("Quantization:          4-bit NF4")
    print("")
    print("── .env / Config ─────────────────────────────────────")
    print(f"  MAX_SEQ_LENGTH:      {Config.MAX_SEQ_LENGTH}")
    print(f"  QLORA_EPOCHS:        {Config.QLORA_EPOCHS}")
    print(f"  QLORA_BATCH_SIZE:    {Config.QLORA_BATCH_SIZE}")
    print(f"  QLORA_GRAD_ACCUM:    {Config.QLORA_GRAD_ACCUM}")
    print(f"  QLORA_LR:            {Config.QLORA_LR}")
    print(f"  DORA_RANK:           {dora_rank}")
    print(f"  DORA_FULL_MODULES:   {'1 (attn + MLP)' if dora_full else '0 (attn only)'}")
    print("──────────────────────────────────────────────────────")
    print(f"  Effective Batch:     {effective_batch}  ({Config.QLORA_BATCH_SIZE} × {Config.QLORA_GRAD_ACCUM} grad accum)")
    print(f"  Training Examples:   {len(dataset)}")
    print(f"  Total Steps:         {len(dataset) // effective_batch * Config.QLORA_EPOCHS}")
    
    logger.info(f"Starting DoRA training: {Config.QLORA_EPOCHS} epochs, {len(dataset)} examples")
    
    # Train
    trainer.train()
    
    logger.info("Training completed")
    return trainer


def save_model(model, tokenizer, logger):
    """Save the trained DoRA model."""
    output_path = Config.MODELS_DIR / "dora_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save DoRA adapters
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Model saved to {output_path}")
    logger.info(f"Model saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print_header("SCRIPT 4: DORA FINE-TUNING (4-bit Quantized)")
    
    # Setup
    logger = setup_logging("4_finetune")
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

        # Pick the most aggressive tier the GPU's total VRAM can support.
        # Rank + module set are fixed here; seq_len can be reduced on OOM.
        auto_configure_vram(logger, reduce=False)

        # Apply DoRA (rank is now fixed for the lifetime of this run)
        print("\nApplying DoRA adapters...")
        print("DoRA decomposes weights into magnitude and direction for better performance")
        dora_config = create_dora_config()
        model = get_peft_model(model, dora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        model.print_trainable_parameters()
        print(f"\nTrainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # ── Training retry loop ────────────────────────────────────────────────
        # Start at the selected tier; if CUDA OOMs, drop seq_len one tier and
        # re-tokenize.  Rank/modules stay fixed (would need a model reload to
        # change them).  Loop ends on success or when no lower tier exists.
        trained = False
        while not trained:
            dataset = prepare_dataset(texts, tokenizer, logger)
            try:
                train_model(model, tokenizer, dataset, logger)
                trained = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning(
                    f"OOM at MAX_SEQ_LENGTH={Config.MAX_SEQ_LENGTH} — trying lower tier"
                )
                print(f"\n⚠ OOM at seq_len={Config.MAX_SEQ_LENGTH}. Dropping one tier...")
                if not auto_configure_vram(logger, reduce=True):
                    logger.error("OOM and no lower tier available — giving up")
                    print("\n✗ Out of memory even at the lowest tier.")
                    print("  Try setting QLORA_BATCH_SIZE=1 in .env, or use a GPU with more VRAM.")
                    sys.exit(1)

        # Save model
        save_model(model, tokenizer, logger)

        vram = get_vram_usage()
        print(f"\nFinal VRAM Usage: {vram['allocated']:.2f} GB")

        print_header("✓ Script 4 completed successfully!")
        logger.info("Script completed successfully")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

