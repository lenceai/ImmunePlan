#!/usr/bin/env python3
"""
Step 05: Adaptive Fine-Tuning
Book: Chapter 5 — Auto-select QLoRA/DoRA/High-Rank LoRA based on available GPU memory.

Training adapts to hardware:
  < 16 GB VRAM → QLoRA (rank 8, attention only)
  16-23 GB     → QLoRA (rank 16, attention only)
  24-39 GB     → DoRA (rank 32, attention + MLP)
  40+ GB       → High-Rank LoRA (rank 256, all layers)

Standalone: python pipeline/05_finetune.py
Output:     models/finetuned_model/
"""
import sys
import gc
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, load_json, save_json,
    get_vram_gb, select_training_tier, clear_gpu, get_quantization_config,
    MODEL_NAME, DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, TRAIN_EPOCHS,
)


def load_training_data():
    data_file = DATA_DIR / "papers_training_data.json"
    if not data_file.exists():
        print(f"Training data not found: {data_file}")
        print("Run step 03 first.")
        sys.exit(1)
    data = load_json(data_file)
    return [item['text'] for item in data]


def run():
    print_step(5, "ADAPTIVE FINE-TUNING")
    logger = setup_logging("05_finetune")
    ensure_directories()

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset

    vram = get_vram_gb()
    if vram == 0:
        print("No GPU detected. Fine-tuning requires CUDA.")
        print("Skipping — use baseline model or run on GPU-equipped machine.")
        return {"status": "skipped", "reason": "no_gpu"}

    tier_name, tier = select_training_tier(vram)
    print(f"GPU: {vram:.1f} GB VRAM")
    print(f"Selected tier: {tier_name}")
    print(f"  Method: {tier['method']}")
    print(f"  Rank: {tier['rank']}, Alpha: {tier['alpha']}")
    print(f"  Targets: {tier['targets']}")
    print(f"  Batch: {tier['batch_size']}, Grad accum: {tier['grad_accum']}")
    print(f"  Seq length: {tier['seq_length']}")
    logger.info(f"Tier: {tier_name} ({tier['method']}, rank={tier['rank']}, vram={vram:.1f}GB)")

    texts = load_training_data()
    print(f"\nTraining examples: {len(texts)}")
    if len(texts) < 10:
        print("WARNING: Very few training examples. Results may be poor.")

    # --- Load model ---
    print(f"\nLoading model: {MODEL_NAME}...")
    clear_gpu()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=get_quantization_config(),
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    # --- Apply LoRA/DoRA ---
    lora_config = LoraConfig(
        r=tier['rank'], lora_alpha=tier['alpha'],
        target_modules=tier['targets'], lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        use_dora=tier.get('use_dora', False),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Prepare dataset ---
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=tier['seq_length'], padding=False)

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")
    tokenized = tokenized.map(lambda x: {"length": len(x["input_ids"])}, desc="Lengths")

    # --- Train ---
    output_dir = CHECKPOINTS_DIR / f"{tier_name}_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = tier['batch_size'] * tier['grad_accum']
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=tier['batch_size'],
        gradient_accumulation_steps=tier['grad_accum'],
        learning_rate=tier['lr'],
        bf16=True,
        logging_steps=10,
        logging_dir=str(LOGS_DIR / "tensorboard"),
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

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\nStarting {tier['method']} training...")
    print(f"  Epochs: {TRAIN_EPOCHS}, Effective batch: {effective_batch}")
    print(f"  Steps: ~{len(tokenized) // effective_batch * TRAIN_EPOCHS}")
    logger.info(f"Training start: {tier['method']}, {len(texts)} examples")

    trainer.train()

    # --- Save ---
    model_path = MODELS_DIR / "finetuned_model"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    save_json({"tier": tier_name, "method": tier['method'], "rank": tier['rank'],
               "vram_gb": vram, "examples": len(texts), "epochs": TRAIN_EPOCHS},
              model_path / "training_info.json")

    print(f"\nModel saved to {model_path}")
    logger.info("Training complete")
    return {"tier": tier_name, "method": tier['method'], "model_path": str(model_path)}


def main():
    try:
        run()
    except Exception as e:
        if "OutOfMemory" in str(type(e).__name__):
            print(f"\nOut of memory! Try reducing MAX_SEQ_LENGTH or increasing GPU memory.")
        else:
            raise

if __name__ == "__main__":
    main()
