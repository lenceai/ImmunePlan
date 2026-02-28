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
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num:02d}: {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
# BENCHMARK QUESTIONS
# =============================================================================

# Gold-standard reference answers — used for RAGAS answer_correctness metric.
# One concise clinical summary per question ID; grounded in published guidelines.
AUTOIMMUNE_GROUND_TRUTH = {
    "RA001": (
        "The 2010 ACR/EULAR classification criteria score joint involvement (0-5), serology RF/anti-CCP (0-3), "
        "acute-phase reactants CRP/ESR (0-1), and symptom duration (0-1); ≥6/10 classifies RA. "
        "RF sensitivity ~70%, specificity ~85%; anti-CCP sensitivity ~67%, specificity >95%. "
        "Anti-CCP positivity predicts erosive, progressive disease and justifies early aggressive therapy."
    ),
    "RA002": (
        "Anti-CCP 120 U/mL with bilateral wrist synovitis >6 weeks strongly suggests early seropositive RA "
        "(2010 ACR/EULAR score likely ≥6). MRI detects subclinical synovitis before erosions appear. "
        "Recommended workup: baseline X-rays hands/feet, MRI wrists, CBC, LFTs, hepatitis serology before MTX. "
        "Treat-to-target remission (DAS28 <2.6) should begin within 3 months of symptom onset."
    ),
    "RA003": (
        "Anti-CCP >200 U/mL with DAS28 5.8 indicates high disease activity, seropositive erosive RA with poor prognosis. "
        "ACR/EULAR guidelines recommend triple-therapy DMARDs (MTX + SSZ + HCQ) or MTX monotherapy as first-line. "
        "Add biologic (TNF inhibitor or IL-6 blocker) if DAS28 remains >3.2 after 3-6 months of MTX optimisation. "
        "Target DAS28 <2.6 (remission) or <3.2 (low disease activity) with monthly reassessment."
    ),
    "CD001": (
        "Crohn's disease diagnosis requires endoscopic, histological, and radiological evidence of transmural inflammation. "
        "Skip lesions, cobblestone mucosa, and aphthous ulcers on ileocolonoscopy are hallmarks. "
        "Fecal calprotectin >150 µg/g and CRP >5 mg/L indicate active mucosal inflammation. "
        "Workup: MRI enterography for small bowel extent, ASCA/ANCA serology, histology, stool cultures to exclude infection."
    ),
    "CD002": (
        "Fecal calprotectin 680 µg/g, CRP 52 mg/L, and positive ASCA with ileocolonic involvement indicate severe Crohn's "
        "with high risk of complications and surgery. First-line biologic: anti-TNF (infliximab preferred for rapid onset "
        "in severe disease) or ustekinumab (IL-12/23 inhibitor). Combination with immunomodulator reduces immunogenicity. "
        "Target clinical remission (CDAI <150) and mucosal healing within 6-12 months."
    ),
    "CD003": (
        "For moderate-to-severe Crohn's (CDAI 280) failing budesonide and mesalamine, step-up to biologic therapy is indicated. "
        "Anti-TNF agents (infliximab 5 mg/kg IV or adalimumab 160/80/40 mg) achieve 30-40% remission rates. "
        "Ileocolonic disease without perianal complications: infliximab or vedolizumab are first choices. "
        "Combination with azathioprine or MTX reduces antibody formation and improves sustained remission."
    ),
    "REM001": (
        "Biomarkers predicting first-line remission: high anti-CCP titre and elevated CRP favour biologic over conventional DMARDs in RA. "
        "ASCA+/ANCA- pattern, elevated calprotectin, and isolated ileal Crohn's favour anti-TNF as first biologic. "
        "Patient factors: MTX contraindicated in hepatic disease (prefer leflunomide); pregnancy planned (prefer certolizumab). "
        "Treat-to-target with monthly DAS28/CDAI monitoring and therapy adjustment within 3 months of initiating a new agent."
    ),
    "Q001": (
        "Malar rash, photosensitivity, oral ulcers, and polyarthritis with ANA 1:640 and elevated anti-dsDNA meet ≥4 "
        "of 11 ACR criteria (or SLICC 2012 criteria) for SLE. Anti-dsDNA antibodies are highly specific (>95%) and correlate "
        "with lupus nephritis activity. Priority workup: urinalysis/protein-creatinine ratio, complement C3/C4, anti-Sm, anti-phospholipid panel, CBC for cytopenias."
    ),
    "Q002": (
        "Sjogren's syndrome is classified by 2016 ACR/EULAR criteria: lip biopsy focal lymphocytic sialadenitis score ≥1 foci/4mm² (weight 3), "
        "anti-SSA/Ro positive (weight 3), Schirmer's ≤5 mm/5 min (weight 1), ocular staining score ≥5 (weight 1), unstimulated salivary flow ≤0.1 mL/min (weight 1). "
        "Systemic complications: interstitial lung disease, peripheral neuropathy, B-cell lymphoma (44× increased risk), vasculitis, renal tubular acidosis."
    ),
    "Q003": (
        "Fatigue, arthralgia, positive ANA with low complement suggests SLE. Concurrent hyperthyroidism may represent Hashimoto's thyroiditis (polyautoimmunity). "
        "Vitiligo indicates additional organ-specific autoimmunity (autoimmune polyglandular syndrome). "
        "Differential: SLE with secondary organ-specific disease vs MCTD vs undifferentiated connective tissue disease. "
        "Key tests: anti-dsDNA, anti-Sm, anti-Ro/La, anti-U1-RNP, TSH, free T4, anti-TPO antibodies, complete ANA ENA panel."
    ),
}


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
