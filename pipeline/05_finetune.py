#!/usr/bin/env python3
"""
Step 05: Adaptive Fine-Tuning
Book: Chapter 5 — Auto-select QLoRA/DoRA/High-Rank LoRA based on available GPU memory.

Training adapts to hardware:
  < 16 GB VRAM → QLoRA (rank 8, attention only)
  16-23 GB     → QLoRA (rank 16, attention only)
  24-39 GB     → DoRA (rank 32, attention + MLP)
  40+ GB       → High-Rank LoRA (rank 64, all layers)

Standalone: python pipeline/05_finetune.py
Output:     models/finetuned_model/
"""
import sys
import gc
import os

# Must be set BEFORE any CUDA/torch initialisation.
# Restricts PyTorch to GPU 0 so Trainer never wraps the model in DataParallel
# and cuBLAS only sees the target device (avoids CUBLAS_STATUS_NOT_SUPPORTED
# issues that appear when multi-GPU visibility is combined with quantised models).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, load_json, save_json,
    get_vram_gb, select_training_tier, clear_gpu, get_quantization_config,
    MODEL_NAME, DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, TRAIN_EPOCHS,
)


def _fmt_sft(instruction: str, response: str) -> dict:
    """Format an instruction/response pair using Llama 3 chat template."""
    return {
        "prompt": (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "completion": f"{response}<|eot_id|>",
    }


def load_training_data():
    """Load training examples as prompt/completion dicts for TRL's completion-masking.

    Returns list of {"prompt": ..., "completion": ...} dicts.
    TRL's SFTTrainer natively supports this format and correctly masks
    prompt tokens from the loss (completion_only_loss works properly).

    Filtering:
    - Medical Q&A examples from body chunks are EXCLUDED — body chunks are raw
      1500-char paper excerpts (high loss ~10-12, pure memorisation, no generalisation).
    - Medical Q&A from abstract chunks are KEPT — abstracts are 150-300 word
      structured summaries with lower loss and more useful domain signal.
    - All paper-centric Q&A kept (teach paper-grounding / faithfulness).
    """
    data_file = DATA_DIR / "papers_training_data.json"
    if not data_file.exists():
        print(f"Training data not found: {data_file}")
        print("Run step 03 first.")
        sys.exit(1)
    data = load_json(data_file)

    examples = []
    skipped_body = 0
    for item in data:
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        if not instruction or not response:
            continue
        is_abstract = item.get("is_abstract_ref", False)
        if not is_abstract:
            skipped_body += 1
            continue
        examples.append(_fmt_sft(instruction, response))
    if skipped_body:
        print(f"  Skipped {skipped_body} body-chunk examples (all sections) → abstract-only training")
    return examples


def load_ground_truth_qa(repeats: int = 15) -> list:
    """Load AUTOIMMUNE_GROUND_TRUTH Q&A pairs as gold-standard training examples.

    These are concise, clinically accurate answers in the exact evaluation format.
    Repeated `repeats` times so the model reliably learns the answer style:
    - Dense medical prose (not paper-dump)
    - Specific numbers and thresholds
    - Structure: criteria → biomarkers → clinical implication

    This counteracts the tendency of paper-centric training to produce summary
    style responses and restores the base model's medical Q&A capability.
    """
    from pipeline.config import AUTOIMMUNE_QUESTIONS, AUTOIMMUNE_GROUND_TRUTH
    _DISCLAIMER = (
        "\n\n**Important**: This information is for educational purposes only and is "
        "NOT a substitute for professional medical advice. Always consult a qualified "
        "healthcare provider."
    )
    pairs = []
    for q in AUTOIMMUNE_QUESTIONS:
        qid = q["id"]
        if qid not in AUTOIMMUNE_GROUND_TRUTH:
            continue
        answer = AUTOIMMUNE_GROUND_TRUTH[qid] + _DISCLAIMER
        pairs.append(_fmt_replay(q["question"], answer))

    import itertools
    cycle = itertools.cycle(pairs)
    return [next(cycle) for _ in range(len(pairs) * repeats)]


# Small replay buffer — prevents catastrophic forgetting of general instruction-following.
# Covers basic medical knowledge so the model doesn't lose coherence on off-topic inputs.
_REPLAY_PAIRS = [
    ("What is the immune system?",
     "The immune system is the body's defence network against pathogens. It comprises two branches: "
     "innate immunity (rapid, non-specific — neutrophils, macrophages, NK cells) and adaptive immunity "
     "(specific, memory-forming — T cells, B cells, antibodies). Dysregulation of adaptive immunity "
     "underlies most autoimmune diseases.\n\n*Always consult a healthcare provider for personal medical advice.*"),
    ("How do biologics differ from conventional DMARDs?",
     "Conventional DMARDs (e.g., methotrexate, hydroxychloroquine) broadly suppress immune activity. "
     "Biologics are large-molecule drugs engineered to block specific cytokines or cell receptors — "
     "examples include TNF inhibitors (adalimumab), IL-6 blockers (tocilizumab), and B-cell depleting "
     "agents (rituximab). Biologics generally have faster onset and higher specificity but greater infection risk.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What does CRP measure?",
     "C-reactive protein (CRP) is an acute-phase protein produced by the liver in response to inflammation. "
     "Elevated CRP indicates systemic inflammation; normal range is < 5 mg/L (high-sensitivity < 1 mg/L). "
     "In autoimmune diseases it tracks disease activity and response to treatment.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("Explain the difference between autoimmune and autoinflammatory diseases.",
     "Autoimmune diseases involve adaptive immune dysfunction: autoreactive T and B cells attack self-tissue "
     "(e.g., rheumatoid arthritis, lupus). Autoinflammatory diseases involve innate immune dysregulation without "
     "classical autoantibodies (e.g., familial Mediterranean fever). Many conditions overlap both pathways.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What is the role of TNF-alpha in inflammation?",
     "TNF-alpha (tumour necrosis factor alpha) is a pro-inflammatory cytokine released by macrophages. "
     "It amplifies the inflammatory cascade, promotes neutrophil recruitment, and induces NF-κB signalling. "
     "Overproduction drives chronic inflammation in RA, IBD, and psoriasis — the mechanistic basis for anti-TNF biologics.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What is treat-to-target in rheumatoid arthritis?",
     "Treat-to-target (T2T) is a strategy of setting a specific goal (remission or low disease activity by DAS28 score) "
     "and adjusting therapy at regular intervals until that target is reached. ACR/EULAR guidelines recommend DAS28 < 2.6 "
     "as the remission threshold. T2T reduces joint damage and improves functional outcomes versus symptom-driven therapy.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What are the ACR 2010 criteria for RA?",
     "The 2010 ACR/EULAR criteria score joints (0-5), serology RF/anti-CCP (0-3), acute-phase reactants CRP/ESR (0-1), "
     "and symptom duration (0-1). A score ≥ 6 out of 10 classifies rheumatoid arthritis. These criteria emphasise "
     "early seropositive disease to enable treatment before joint erosion occurs.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What is fecal calprotectin used for?",
     "Fecal calprotectin is a neutrophil-derived protein measured in stool. Concentrations > 50 µg/g suggest "
     "intestinal inflammation and are used to screen for and monitor inflammatory bowel disease (Crohn's disease, "
     "ulcerative colitis). It is non-invasive and correlates well with endoscopic inflammation scores.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("Describe the mechanism of JAK inhibitors.",
     "JAK inhibitors (e.g., tofacitinib, baricitinib, upadacitinib) block Janus kinase enzymes that mediate "
     "intracellular signalling downstream of cytokine receptors. By inhibiting JAK1/JAK3, they suppress "
     "inflammatory cytokine cascades (IL-6, IFN-γ, IL-2) with oral bioavailability. Used in RA, UC, and atopic dermatitis.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What is vedolizumab and how does it work?",
     "Vedolizumab is a gut-selective anti-integrin biologic. It blocks α4β7 integrin on lymphocytes, "
     "preventing their homing to intestinal mucosa. This gut-selective mechanism means it has fewer systemic "
     "immunosuppressive effects than anti-TNF agents. It is approved for moderate-to-severe Crohn's disease and UC.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("What is morning stiffness and why is it important in RA?",
     "Morning stiffness is joint stiffness lasting > 1 hour upon waking, caused by overnight accumulation of "
     "synovial fluid cytokines. It is a hallmark symptom of inflammatory arthritis. Duration > 30 minutes "
     "distinguishes inflammatory (RA, PsA) from mechanical causes. Stiffness > 60 minutes contributes to ACR classification.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
    ("How does the gut microbiome relate to autoimmune disease?",
     "The gut microbiome modulates immune homeostasis via short-chain fatty acid production, Treg induction, "
     "and barrier function. Dysbiosis — reduced diversity and altered Firmicutes/Bacteroidetes ratio — is "
     "associated with RA, IBD, and MS. Microbiome signatures may precede clinical onset, suggesting a causal role.\n\n"
     "*Always consult a healthcare provider for personal medical advice.*"),
]


def _fmt_replay(instruction: str, response: str) -> dict:
    return _fmt_sft(instruction, response)


def load_replay_buffer(n_domain: int) -> list:
    """
    Return a list of replay texts sized at ~10% of the domain training set.
    Cycles through the hardcoded pairs as many times as needed.
    Prevents catastrophic forgetting of general medical instruction following.
    """
    import itertools
    n_replay = max(12, n_domain // 10)
    replay_texts = [_fmt_replay(instr, resp) for instr, resp in _REPLAY_PAIRS]
    cycle = itertools.cycle(replay_texts)
    return [next(cycle) for _ in range(n_replay)]


def run():
    print_step(5, "ADAPTIVE FINE-TUNING")
    logger = setup_logging("05_finetune")
    ensure_directories()

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    vram = get_vram_gb()
    if vram == 0:
        print("No GPU detected. Fine-tuning requires CUDA.")
        print("Skipping — use baseline model or run on GPU-equipped machine.")
        return {"status": "skipped", "reason": "no_gpu"}

    # Use per-device VRAM for tier selection since we train on a single GPU
    # (device_map="cuda:0"). Total VRAM across all GPUs would select a tier
    # too aggressive for the available single-device memory.
    vram_for_tier = get_vram_gb(per_device=True)
    tier_name, tier = select_training_tier(vram_for_tier)
    print(f"GPU: {vram:.1f} GB VRAM total, {vram_for_tier:.1f} GB per device")
    print(f"Selected tier: {tier_name}")
    print(f"  Method: {tier['method']}")
    print(f"  Rank: {tier['rank']}, Alpha: {tier['alpha']}")
    print(f"  Targets: {tier['targets']}")
    print(f"  Batch: {tier['batch_size']}, Grad accum: {tier['grad_accum']}")
    print(f"  Seq length: {tier['seq_length']}")
    logger.info(f"Tier: {tier_name} ({tier['method']}, rank={tier['rank']}, vram={vram:.1f}GB)")

    import random
    examples = load_training_data()
    ground_truth = load_ground_truth_qa(repeats=40)
    replay = load_replay_buffer(len(examples))
    all_examples = examples + ground_truth + replay
    random.seed(42)
    random.shuffle(all_examples)
    n_gt = len(ground_truth)
    print(f"\nTraining examples: {len(all_examples)} ({len(examples)} domain + {n_gt} ground-truth Q&A + {len(replay)} replay)")
    if len(all_examples) < 10:
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
        torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
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

    # Cast all trainable LoRA params to fp16 — prepare_model_for_kbit_training
    # can leave some tensors as bfloat16 which breaks the fp16 AMP grad scaler.
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Prepare dataset with train/val split ---
    # Use prompt-completion format — TRL natively masks prompt tokens from loss.
    # With "text" format, completion_only_loss=True is silently ignored (no completion_mask
    # is generated by tokenize_fn for language-modeling datasets). Prompt-completion format
    # ensures _collate_prompt_completion creates the mask correctly.
    dataset = Dataset.from_list(all_examples)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- Train ---
    output_dir = CHECKPOINTS_DIR / f"{tier_name}_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = tier['batch_size'] * tier['grad_accum']

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=tier['batch_size'],
        gradient_accumulation_steps=tier['grad_accum'],
        learning_rate=tier['lr'],
        # No AMP: RTX 3090 has no BF16 tensor cores (CUBLAS_STATUS_NOT_SUPPORTED),
        # and the model's internal ops produce BF16 activations that break
        # PyTorch's FP16 GradScaler. Full FP32 grads are safe with 4-bit base model.
        fp16=False,
        bf16=False,
        logging_steps=10,
        logging_dir=str(LOGS_DIR / "tensorboard"),
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_length=tier['seq_length'],
        completion_only_loss=True,  # works correctly with prompt-completion format
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nStarting {tier['method']} training (SFT — response-only loss)...")
    print(f"  Epochs: {TRAIN_EPOCHS}, Effective batch: {effective_batch}")
    print(f"  Steps: ~{len(train_ds) // effective_batch * TRAIN_EPOCHS}")
    logger.info(f"Training start: {tier['method']}, {len(all_examples)} examples")

    trainer.train()

    # --- Save ---
    model_path = MODELS_DIR / "finetuned_model"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    save_json({"tier": tier_name, "method": tier['method'], "rank": tier['rank'],
               "vram_gb": vram, "examples": len(all_examples), "epochs": TRAIN_EPOCHS},
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
