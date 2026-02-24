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

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/Nemotron-Cascade-8B-Thinking")
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
        "method": "High-Rank LoRA", "rank": 256, "alpha": 512,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
        "use_dora": False, "batch_size": 1, "grad_accum": 32,
        "seq_length": 512, "lr": 5e-5,
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


def get_vram_gb() -> float:
    info = check_cuda()
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
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(model_name=None, quantize=True, device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_map, "trust_remote_code": True}
    if quantize:
        kwargs["quantization_config"] = get_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def format_prompt(question, system_prompt=None, tokenizer=None, use_thinking=True):
    model_lower = MODEL_NAME.lower()

    if "nemotron" in model_lower or "cascade" in model_lower:
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            ctrl = "/think" if use_thinking else "/no_think"
            messages.append({"role": "user", "content": f"{ctrl} {question}"})
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        ctrl = "/think" if use_thinking else "/no_think"
        if system_prompt:
            return f"<|system|>\n{system_prompt}<|user|>\n{ctrl} {question}<|assistant|>\n"
        return f"<|user|>\n{ctrl} {question}<|assistant|>\n"

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
                      temperature=0.7, top_p=0.9, do_sample=True):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full[len(prompt):].strip()
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
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num:02d}: {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
# BENCHMARK QUESTIONS
# =============================================================================

AUTOIMMUNE_QUESTIONS = [
    {"id": "RA001", "category": "Rheumatoid Arthritis - Early Diagnosis", "difficulty": "medium",
     "question": "A 45-year-old patient has symmetric polyarthritis affecting wrists, MCPs, and PIPs for 6 months, with morning stiffness lasting over 1 hour. RF and anti-CCP antibodies are positive. Explain the diagnostic criteria for rheumatoid arthritis and the significance of these serological markers for early diagnosis."},
    {"id": "RA002", "category": "Rheumatoid Arthritis - Early Diagnosis", "difficulty": "medium",
     "question": "A 38-year-old woman presents with 3 months of bilateral wrist pain and morning stiffness lasting 45 minutes. X-rays show no erosions. Anti-CCP is positive at 120 U/mL, RF is negative, ESR is 28 mm/h, CRP is 12 mg/L. What is the likelihood this is early RA, and what additional tests or monitoring would you recommend to confirm diagnosis before joint damage occurs?"},
    {"id": "RA003", "category": "Rheumatoid Arthritis - Biomarkers", "difficulty": "hard",
     "question": "A newly diagnosed RA patient has high anti-CCP titers (>200 U/mL), positive RF, elevated CRP (45 mg/L), and high disease activity score (DAS28 = 5.8). What do these biomarkers predict about disease course, and which single treatment approach would you recommend to achieve remission most efficiently?"},
    {"id": "CD001", "category": "Crohn's Disease - Early Diagnosis", "difficulty": "medium",
     "question": "A 25-year-old patient presents with 4 months of abdominal pain, diarrhea (5-6 loose stools/day), weight loss of 8 kg, and perianal fistulas. Fecal calprotectin is 450 ug/g, CRP is 35 mg/L. Colonoscopy shows patchy inflammation in terminal ileum and right colon with skip lesions. What diagnostic criteria confirm Crohn's disease, and what additional tests are needed for complete assessment?"},
    {"id": "CD002", "category": "Crohn's Disease - Biomarkers", "difficulty": "hard",
     "question": "A newly diagnosed Crohn's patient has elevated fecal calprotectin (680 ug/g), high CRP (52 mg/L), positive ASCA antibodies, and extensive ileocolonic disease on imaging. What do these biomarkers indicate about disease severity and prognosis? Which single treatment would you select to achieve remission based on these markers?"},
    {"id": "CD003", "category": "Crohn's Disease - Treatment Selection", "difficulty": "very_hard",
     "question": "A 30-year-old with moderate-to-severe Crohn's disease (CDAI = 280) has failed mesalamine and budesonide. They have ileocolonic disease, no perianal complications, and elevated CRP. What single biologic therapy would you choose to maximize first-line remission rates, and what patient factors or biomarkers guide this decision?"},
    {"id": "REM001", "category": "Remission Strategy", "difficulty": "very_hard",
     "question": "Current autoimmune disease treatment often requires sequential trials of 3-5 different medications before achieving remission. What factors (biomarkers, disease characteristics, patient factors) should guide initial treatment selection to maximize the probability of achieving remission with a single first-line therapy?"},
    {"id": "Q001", "category": "Systemic Lupus Erythematosus", "difficulty": "medium",
     "question": "A 28-year-old woman presents with malar rash, photosensitivity, oral ulcers, and joint pain. Her ANA is positive at 1:640 with a speckled pattern, and anti-dsDNA antibodies are elevated. What is the most likely diagnosis, and what are the key diagnostic criteria you would use?"},
    {"id": "Q002", "category": "Sjogren's Syndrome", "difficulty": "hard",
     "question": "A 52-year-old woman complains of dry eyes and dry mouth for 2 years. Schirmer's test is abnormal, and she has positive anti-SSA/Ro and anti-SSB/La antibodies. What diagnostic criteria would you use to confirm Sjogren's syndrome, and what are the potential systemic complications?"},
    {"id": "Q003", "category": "Differential Diagnosis", "difficulty": "very_hard",
     "question": "A 35-year-old woman has fatigue, joint pain, positive ANA, and low complement levels. However, she also has hyperthyroidism and vitiligo. How would you differentiate between multiple autoimmune conditions versus a single systemic autoimmune disease?"},
]
