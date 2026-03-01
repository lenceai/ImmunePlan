#!/usr/bin/env python3
"""
Unified Configuration — Single source of truth for the entire pipeline.

Merges:
  - Reliability specification (Ch 1)
  - Model & training config
  - RAG, safety, monitoring config
  - GPU utilities
  - Model loading & generation utilities
  - Benchmark questions
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


def ensure_directories():
    for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, VECTOR_STORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIG
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "aaditya/Llama3-OpenBioLLM-8B")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")

# Training defaults (overridden by auto GPU selection in step 05)
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "3"))
TRAIN_LR = float(os.getenv("TRAIN_LR", "2e-4"))


# =============================================================================
# RELIABILITY SPEC (Ch 1)
# =============================================================================

RELIABILITY_SPEC = {
    "project": "ImmunePlan",
    "tier": "clinical_support",
    "definition": (
        "Consistently useful, grounded in medical literature, "
        "transparent about uncertainty, safe for patient-facing use."
    ),
    "quality_targets": {
        "groundedness_min": 0.8,
        "hallucination_rate_max": 0.05,
        "citation_rate_min": 0.9,
        "safety_pass_rate_min": 0.99,
    },
    "forbidden_actions": [
        "prescribe_medication", "order_tests", "modify_patient_records",
        "provide_definitive_diagnosis", "override_physician_judgment",
    ],
    "medical_disclaimer": (
        "IMPORTANT: This information is for educational purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, "
        "or treatment. Always consult your healthcare provider."
    ),
}


# =============================================================================
# GPU-ADAPTIVE TRAINING TIERS (Ch 5)
# =============================================================================

TRAINING_TIERS = {
    "minimal": {
        "method": "QLoRA", "rank": 8, "alpha": 16,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "use_dora": False, "batch_size": 1, "grad_accum": 16,
        "seq_length": 512, "lr": 2e-4,
        "vram_range": "< 16 GB",
    },
    "standard": {
        "method": "QLoRA", "rank": 16, "alpha": 32,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "use_dora": False, "batch_size": 2, "grad_accum": 8,
        "seq_length": 1024, "lr": 2e-4,
        "vram_range": "16-23 GB",
    },
    "enhanced": {
        "method": "DoRA", "rank": 32, "alpha": 64,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
        "use_dora": True, "batch_size": 2, "grad_accum": 8,
        "seq_length": 2048, "lr": 2e-4,
        "vram_range": "24-39 GB",
    },
    "maximum": {
        # Fixed: was seq_length=512 which wasted the extra VRAM headroom.
        # At 40+ GB we can run rank-64 with full 2048 context and batch_size=2.
        # Rank-256 with 512 context is worse than rank-64 with 2048 context.
        "method": "High-Rank LoRA", "rank": 64, "alpha": 128,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
        "use_dora": False, "batch_size": 2, "grad_accum": 16,
        "seq_length": 2048, "lr": 5e-5,
        "vram_range": "40+ GB",
    },
}


def select_training_tier(vram_gb: float) -> Tuple[str, Dict]:
    if vram_gb >= 40:
        return "maximum", TRAINING_TIERS["maximum"]
    if vram_gb >= 24:
        return "enhanced", TRAINING_TIERS["enhanced"]
    if vram_gb >= 16:
        return "standard", TRAINING_TIERS["standard"]
    return "minimal", TRAINING_TIERS["minimal"]


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(name: str) -> logging.Logger:
    ensure_directories()
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# GPU UTILITIES
# =============================================================================

def check_cuda() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "total_vram_gb": 0, "devices": []}

    info = {"available": torch.cuda.is_available(), "total_vram_gb": 0, "devices": []}
    if info["available"]:
        info["version"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1024**3
            info["devices"].append({"index": i, "name": props.name, "vram_gb": round(gb, 2)})
            info["total_vram_gb"] += gb
        info["total_vram_gb"] = round(info["total_vram_gb"], 2)
    return info


def get_vram_gb(per_device: bool = False) -> float:
    """Return total VRAM or per-device VRAM (min across GPUs).

    Use per_device=True for tier selection when training on a single GPU
    (device_map='cuda:0') to avoid selecting a tier designed for multi-GPU
    total VRAM.
    """
    info = check_cuda()
    if per_device and info.get("devices"):
        return min(d["vram_gb"] for d in info["devices"])
    return info["total_vram_gb"]


def clear_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    # RTX 3090 (SM 8.6) does not support BF16 tensor-core GEMM (CUBLAS_STATUS_NOT_SUPPORTED).
    # Use float16 which is fully supported on Ampere consumer GPUs.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(model_name=None, quantize=True, device_map="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.float16, "device_map": device_map, "trust_remote_code": True}
    if quantize:
        kwargs["quantization_config"] = get_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def format_prompt(question, system_prompt=None, tokenizer=None, **kwargs):
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {question}\n\nAssistant:"
    return f"User: {question}\n\nAssistant:"


def generate_response(model, tokenizer, prompt, max_new_tokens=1024,
                      temperature=0.7, top_p=0.9, do_sample=True,
                      repetition_penalty=1.0):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    gen_time = time.time() - start

    # Decode only the newly generated tokens — avoids prompt-length offset bugs
    # when skip_special_tokens strips markers that were in the original prompt string.
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    for marker in ["<｜end▁of▁sentence｜>", "</s>", "<|endoftext|>"]:
        if marker in response:
            response = response.split(marker)[0].strip()

    tokens = {"input": input_len, "output": outputs.shape[1] - input_len, "total": outputs.shape[1]}
    return response, gen_time, tokens


# =============================================================================
# FILE UTILITIES
# =============================================================================

def save_json(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_header(title, char="=", width=60):
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_step(step_num, title):
    label = f"{step_num:02d}" if isinstance(step_num, int) else str(step_num).upper()
    print(f"\n{'=' * 60}")
    print(f"  STEP {label}: {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
# BENCHMARK QUESTIONS
# =============================================================================

_BENCHMARK_FILE = PROJECT_ROOT / "pipeline" / "autoimmune_remission_benchmark_100_v1.json"
_benchmark = load_json(_BENCHMARK_FILE) or {}

# Prefer the versioned 100-question remission benchmark if present.
if isinstance(_benchmark, dict) and _benchmark.get("questions") and _benchmark.get("ground_truth"):
    AUTOIMMUNE_QUESTIONS = _benchmark["questions"]
    AUTOIMMUNE_GROUND_TRUTH = _benchmark["ground_truth"]
else:
    # Fallback minimal benchmark (legacy)
    AUTOIMMUNE_GROUND_TRUTH = {
        "RA001": (
            "The 2010 ACR/EULAR classification criteria score joint involvement (0-5), serology RF/anti-CCP (0-3), "
            "acute-phase reactants CRP/ESR (0-1), and symptom duration (0-1); ≥6/10 classifies RA. "
            "RF sensitivity ~70%, specificity ~85%; anti-CCP sensitivity ~67%, specificity >95%. "
            "Anti-CCP positivity predicts erosive, progressive disease and justifies early aggressive therapy."
        ),
    }

    AUTOIMMUNE_QUESTIONS = [
        {"id": "RA001", "category": "Rheumatoid Arthritis - Early Diagnosis", "difficulty": "medium",
         "question": "A 45-year-old patient has symmetric polyarthritis affecting wrists, MCPs, and PIPs for 6 months, with morning stiffness lasting over 1 hour. RF and anti-CCP antibodies are positive. Explain the diagnostic criteria for rheumatoid arthritis and the significance of these serological markers for early diagnosis."},
    ]
