#!/usr/bin/env python3
"""
Step 05b: DPO Fine-Tuning (run after 05_finetune.py)

Applies Direct Preference Optimization on top of the SFT checkpoint to
improve citation quality and reduce vague/template responses.

Preference pairs:
  chosen  — detailed, literature-grounded response from training data
  rejected — vague template with no citations or context

Standalone: conda run -n base python pipeline/05b_dpo.py
Output:     models/dpo_model/
"""
import sys
import gc
import os
import random

# Must be set BEFORE any CUDA/torch initialisation.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, load_json, save_json,
    get_vram_gb, select_training_tier, clear_gpu, get_quantization_config,
    MODEL_NAME, DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
)


# ---------------------------------------------------------------------------
# Rejection templates — vary enough to prevent trivial overfitting
# ---------------------------------------------------------------------------

_REJECTION_TEMPLATES = [
    "Based on available medical literature, this is a complex topic that requires personalised assessment. "
    "Please consult your healthcare provider for guidance specific to your situation.",

    "This question touches on an important medical topic. I recommend discussing it with a specialist "
    "who can review your complete clinical picture and provide tailored advice.",

    "Medical conditions like this are best assessed by a qualified clinician. While general information "
    "exists in the literature, specific recommendations depend on individual patient factors.",

    "There is ongoing research in this area. Current evidence suggests multiple approaches may be "
    "appropriate, but treatment decisions should be made with your healthcare team.",

    "This is an area where clinical guidelines vary. The most appropriate course of action depends "
    "on disease severity, comorbidities, and patient preferences — factors best evaluated by a specialist.",
]


def build_dpo_dataset(training_data: list, max_pairs: int = 2000) -> list:
    """
    Build DPO preference pairs from training examples.

    chosen  = the original grounded response from our training data
    rejected = a randomly selected vague template response

    Returns list of dicts with keys: prompt, chosen, rejected
    """
    pairs = []
    rng = random.Random(42)

    for item in training_data:
        if len(pairs) >= max_pairs:
            break
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        if not instruction or not response:
            continue
        # Only use body chunks (not abstract refs) — they have richer responses
        if item.get("is_abstract_ref", False):
            continue
        # Require a reasonably detailed chosen response
        if len(response) < 200:
            continue

        rejected = rng.choice(_REJECTION_TEMPLATES)
        pairs.append({
            "prompt": instruction,
            "chosen": response,
            "rejected": rejected,
        })

    rng.shuffle(pairs)
    return pairs


def run():
    print_step("5b", "DPO FINE-TUNING")
    logger = setup_logging("05b_dpo")
    ensure_directories()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    vram = get_vram_gb()
    vram_for_tier = get_vram_gb(per_device=True)
    if vram == 0:
        print("No GPU detected. DPO requires CUDA. Skipping.")
        return {"status": "skipped", "reason": "no_gpu"}

    # --- Load training data ---
    data_file = DATA_DIR / "papers_training_data.json"
    if not data_file.exists():
        print("Training data not found. Run step 03 first.")
        sys.exit(1)

    training_data = load_json(data_file)
    pairs = build_dpo_dataset(training_data, max_pairs=2000)
    print(f"DPO preference pairs: {len(pairs)}")
    if len(pairs) < 50:
        print("Too few pairs for DPO. Skipping.")
        return {"status": "skipped", "reason": "insufficient_pairs"}

    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)

    # --- Load SFT model ---
    sft_model_path = MODELS_DIR / "finetuned_model"
    base_model_name = MODEL_NAME

    # Check if SFT adapter exists; if so load base+adapter, otherwise just base
    if (sft_model_path / "adapter_config.json").exists():
        print(f"\nLoading SFT adapter from: {sft_model_path}")
    else:
        print(f"\nNo SFT adapter found at {sft_model_path}. Using base model.")
        sft_model_path = None

    print(f"Loading base model: {base_model_name}")
    clear_gpu()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(
        str(sft_model_path) if sft_model_path else base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_quantization_config(),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    if sft_model_path and (sft_model_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(sft_model_path), is_trainable=True)

    # Cast any bfloat16 trainable params to fp16 for the fp16 AMP grad scaler.
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    # --- DPO config — use a lower rank to keep memory usage reasonable ---
    tier_name, tier = select_training_tier(vram_for_tier)
    dpo_rank = max(8, tier["rank"] // 2)  # half the SFT rank for DPO

    output_dir = CHECKPOINTS_DIR / "dpo_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_len = min(tier["seq_length"], 1024)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tier["grad_accum"],
        learning_rate=5e-6,          # much lower LR than SFT
        fp16=False,   # No AMP: model's internal BF16 ops break FP16 GradScaler
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # DPO-specific
        beta=0.1,               # KL divergence weight
        max_length=max_len,
        loss_type="sigmoid",    # standard DPO loss
    )

    print(f"\nDPO config: rank={dpo_rank}, beta=0.1, max_len={max_len}")
    print(f"  Train pairs: {len(train_ds)}, Eval pairs: {len(eval_ds)}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting DPO training...")
    trainer.train()

    # --- Save ---
    dpo_model_path = MODELS_DIR / "dpo_model"
    dpo_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dpo_model_path)
    tokenizer.save_pretrained(dpo_model_path)

    save_json({
        "base": base_model_name,
        "sft_path": str(sft_model_path) if sft_model_path else None,
        "dpo_rank": dpo_rank,
        "pairs": len(pairs),
        "beta": 0.1,
    }, dpo_model_path / "dpo_info.json")

    print(f"\nDPO model saved to {dpo_model_path}")
    logger.info("DPO training complete")
    return {"model_path": str(dpo_model_path), "pairs": len(pairs)}


def main():
    try:
        run()
    except Exception as e:
        if "OutOfMemory" in str(type(e).__name__):
            print("Out of memory! Try reducing max_length or running on larger GPU.")
        else:
            raise


if __name__ == "__main__":
    main()
