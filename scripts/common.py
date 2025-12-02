#!/usr/bin/env python3
"""
Common utilities and shared code for all scripts.
Version: 1.0.0

This module contains shared functions, constants, and configurations
used across all pipeline scripts to avoid code duplication.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration management."""
    
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
    
    # Directories
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
    LOGS_DIR = Path(os.getenv("LOGS_DIR", "./logs"))
    CHECKPOINTS_DIR = Path("./checkpoints")
    
    # Training configs
    QLORA_EPOCHS = int(os.getenv("QLORA_EPOCHS", "3"))
    QLORA_BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "2"))
    QLORA_LR = float(os.getenv("QLORA_LR", "2e-4"))
    
    FULL_EPOCHS = int(os.getenv("FULL_EPOCHS", "3"))
    FULL_BATCH_SIZE = int(os.getenv("FULL_BATCH_SIZE", "1"))
    FULL_LR = float(os.getenv("FULL_LR", "5e-5"))
    
    # PubMed
    PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories."""
        for dir_path in [cls.DATA_DIR, cls.RESULTS_DIR, cls.MODELS_DIR, 
                         cls.LOGS_DIR, cls.CHECKPOINTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        if not cls.PUBMED_EMAIL or cls.PUBMED_EMAIL == "your.email@example.com":
            warnings.append("PUBMED_EMAIL not configured - PubMed downloads will fail")
        return warnings


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(script_name: str) -> logging.Logger:
    """
    Set up logging for a script.
    
    Args:
        script_name: Name of the script for log file naming
    
    Returns:
        Configured logger instance
    """
    Config.ensure_directories()
    
    log_file = Config.LOGS_DIR / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# GPU UTILITIES
# =============================================================================

def check_cuda() -> Dict[str, Any]:
    """
    Check CUDA availability and return GPU information.
    
    Returns:
        Dictionary with CUDA info
    """
    info = {
        "available": torch.cuda.is_available(),
        "version": None,
        "device_count": 0,
        "devices": [],
        "total_vram_gb": 0
    }
    
    if info["available"]:
        info["version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}"
            }
            info["devices"].append(device_info)
            info["total_vram_gb"] += device_info["total_memory_gb"]
    
    return info


def print_cuda_info(logger: Optional[logging.Logger] = None):
    """Print CUDA information."""
    info = check_cuda()
    
    output = []
    if info["available"]:
        output.append(f"✓ CUDA Available: True")
        output.append(f"✓ CUDA Version: {info['version']}")
        output.append(f"✓ GPU Count: {info['device_count']}")
        for device in info["devices"]:
            output.append(f"✓ GPU {device['index']}: {device['name']} ({device['total_memory_gb']:.2f} GB)")
        output.append(f"✓ Total VRAM: {info['total_vram_gb']:.2f} GB")
    else:
        output.append("✗ CUDA not available - running on CPU")
    
    for line in output:
        if logger:
            logger.info(line)
        else:
            print(line)
    
    return info


def get_vram_usage() -> Dict[str, float]:
    """Get current VRAM usage."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_quantization_config() -> BitsAndBytesConfig:
    """Get 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    quantize: bool = True,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with standard configuration.
    
    Args:
        model_name: Model name/path (defaults to Config.MODEL_NAME)
        quantize: Whether to use 4-bit quantization
        device_map: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or Config.MODEL_NAME
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading kwargs
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    if quantize:
        model_kwargs["quantization_config"] = get_quantization_config()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def format_prompt(question: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt for the DeepSeek model.
    
    Args:
        question: The user's question
        system_prompt: Optional system prompt
    
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        return f"<｜begin▁of▁sentence｜>System: {system_prompt}\n\nUser: {question}\n\nAssistant:"
    return f"<｜begin▁of▁sentence｜>User: {question}\n\nAssistant:"


def extract_response(full_output: str, prompt: str) -> str:
    """
    Extract the model's response from the full output.
    
    Args:
        full_output: Full decoded output including prompt
        prompt: The original prompt
    
    Returns:
        Extracted response text
    """
    response = full_output[len(prompt):].strip()
    
    # Remove any trailing special tokens or markers
    end_markers = ["<｜end▁of▁sentence｜>", "</s>", "<|endoftext|>"]
    for marker in end_markers:
        if marker in response:
            response = response.split(marker)[0].strip()
    
    return response


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> Tuple[str, float, Dict[str, int]]:
    """
    Generate a response from the model.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Tuple of (response_text, generation_time, token_counts)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = extract_response(full_response, prompt)
    
    # Token counts
    output_length = outputs.shape[1]
    token_counts = {
        "input_tokens": input_length,
        "output_tokens": output_length - input_length,
        "total_tokens": output_length
    }
    
    return response, generation_time, token_counts


# =============================================================================
# FILE UTILITIES
# =============================================================================

def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Path) -> Optional[Any]:
    """Load data from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# TIMING DECORATOR
# =============================================================================

def timed(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱ {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# =============================================================================
# BENCHMARK QUESTIONS (Single Source of Truth)
# =============================================================================

AUTOIMMUNE_QUESTIONS = [
    # ============================================================================
    # RHEUMATOID ARTHRITIS - DIAGNOSIS & EARLY DETECTION
    # ============================================================================
    {
        "id": "RA001",
        "category": "Rheumatoid Arthritis - Early Diagnosis",
        "difficulty": "medium",
        "question": "A 45-year-old patient has symmetric polyarthritis affecting wrists, MCPs, and PIPs for 6 months, with morning stiffness lasting over 1 hour. RF and anti-CCP antibodies are positive. Explain the diagnostic criteria for rheumatoid arthritis and the significance of these serological markers for early diagnosis."
    },
    {
        "id": "RA002",
        "category": "Rheumatoid Arthritis - Early Diagnosis",
        "difficulty": "medium",
        "question": "A 38-year-old woman presents with 3 months of bilateral wrist pain and morning stiffness lasting 45 minutes. X-rays show no erosions. Anti-CCP is positive at 120 U/mL, RF is negative, ESR is 28 mm/h, CRP is 12 mg/L. What is the likelihood this is early RA, and what additional tests or monitoring would you recommend to confirm diagnosis before joint damage occurs?"
    },
    {
        "id": "RA003",
        "category": "Rheumatoid Arthritis - Biomarkers",
        "difficulty": "hard",
        "question": "A newly diagnosed RA patient has high anti-CCP titers (>200 U/mL), positive RF, elevated CRP (45 mg/L), and high disease activity score (DAS28 = 5.8). What do these biomarkers predict about disease course, and which single treatment approach would you recommend to achieve remission most efficiently?"
    },
    {
        "id": "RA004",
        "category": "Rheumatoid Arthritis - Treatment Selection",
        "difficulty": "hard",
        "question": "A 50-year-old RA patient with high disease activity (DAS28 = 6.2), positive anti-CCP, and early erosions on X-ray has failed methotrexate monotherapy after 3 months. Based on current evidence, what single biologic or targeted synthetic DMARD would you choose to achieve remission, and what biomarkers would you monitor to predict treatment response?"
    },
    {
        "id": "RA005",
        "category": "Rheumatoid Arthritis - Remission Strategy",
        "difficulty": "very_hard",
        "question": "A patient with early RA (symptom duration <6 months) has high anti-CCP, moderate disease activity, and no erosions. Current guidelines suggest starting with methotrexate, but many patients require 3-5 treatment changes before remission. What single treatment strategy (including specific drug, dose, and monitoring approach) would maximize the chance of achieving remission with the first intervention?"
    },
    {
        "id": "RA006",
        "category": "Rheumatoid Arthritis - Remission Criteria",
        "difficulty": "medium",
        "question": "A RA patient has been on treatment for 6 months. Their DAS28 is 2.1, they have no swollen joints, morning stiffness is <15 minutes, and CRP is normal. However, they still report mild fatigue. Has this patient achieved remission? What are the ACR/EULAR remission criteria, and how would you adjust treatment to maintain remission?"
    },
    {
        "id": "RA007",
        "category": "Rheumatoid Arthritis - Treatment Optimization",
        "difficulty": "very_hard",
        "question": "A RA patient achieved remission after 8 months on adalimumab + methotrexate. They want to reduce treatment burden. Current practice often involves multiple drug combinations. What evidence-based approach would you use to maintain remission with a single agent, and what monitoring strategy would ensure early detection of flare?"
    },
    {
        "id": "RA008",
        "category": "Rheumatoid Arthritis - Seronegative",
        "difficulty": "hard",
        "question": "A patient has symmetric polyarthritis, morning stiffness, and elevated inflammatory markers, but both RF and anti-CCP are negative. How do you diagnose seronegative RA, and does the absence of these antibodies affect your treatment strategy for achieving remission?"
    },
    
    # ============================================================================
    # CROHN'S DISEASE - DIAGNOSIS & EARLY DETECTION
    # ============================================================================
    {
        "id": "CD001",
        "category": "Crohn's Disease - Early Diagnosis",
        "difficulty": "medium",
        "question": "A 25-year-old patient presents with 4 months of abdominal pain, diarrhea (5-6 loose stools/day), weight loss of 8 kg, and perianal fistulas. Fecal calprotectin is 450 μg/g, CRP is 35 mg/L. Colonoscopy shows patchy inflammation in terminal ileum and right colon with skip lesions. What diagnostic criteria confirm Crohn's disease, and what additional tests are needed for complete assessment?"
    },
    {
        "id": "CD002",
        "category": "Crohn's Disease - Biomarkers",
        "difficulty": "hard",
        "question": "A newly diagnosed Crohn's patient has elevated fecal calprotectin (680 μg/g), high CRP (52 mg/L), positive ASCA antibodies, and extensive ileocolonic disease on imaging. What do these biomarkers indicate about disease severity and prognosis? Which single treatment would you select to achieve remission based on these markers?"
    },
    {
        "id": "CD003",
        "category": "Crohn's Disease - Treatment Selection",
        "difficulty": "very_hard",
        "question": "A 30-year-old with moderate-to-severe Crohn's disease (CDAI = 280) has failed mesalamine and budesonide. They have ileocolonic disease, no perianal complications, and elevated CRP. Current practice often requires sequential trials of multiple biologics. What single biologic therapy would you choose to maximize first-line remission rates, and what patient factors or biomarkers guide this decision?"
    },
    {
        "id": "CD004",
        "category": "Crohn's Disease - Remission Strategy",
        "difficulty": "very_hard",
        "question": "A patient with newly diagnosed Crohn's disease has high disease activity, elevated inflammatory markers, and extensive small bowel involvement. Many patients require 3-5 different treatments before achieving remission. What single treatment approach (drug, dose, route, monitoring) would optimize the chance of achieving deep remission (clinical + endoscopic + biomarker) with the first intervention?"
    },
    {
        "id": "CD005",
        "category": "Crohn's Disease - Remission Criteria",
        "difficulty": "medium",
        "question": "A Crohn's patient has been on infliximab for 6 months. They report no abdominal pain, normal stool frequency, CDAI is 120, CRP is normal, and fecal calprotectin is 45 μg/g. Colonoscopy shows mucosal healing. Has this patient achieved remission? What are the criteria for deep remission in Crohn's disease, and how do you maintain it?"
    },
    {
        "id": "CD006",
        "category": "Crohn's Disease - Treatment Optimization",
        "difficulty": "very_hard",
        "question": "A Crohn's patient achieved deep remission on combination therapy (anti-TNF + immunomodulator) but wants to simplify treatment. What evidence supports maintaining remission with a single agent, and what monitoring strategy (clinical, biomarker, endoscopic) would detect early relapse before symptoms recur?"
    },
    {
        "id": "CD007",
        "category": "Crohn's Disease - Perianal Disease",
        "difficulty": "hard",
        "question": "A Crohn's patient has active perianal fistulas along with ileocolonic inflammation. Perianal disease often requires multiple treatment approaches. What single treatment strategy would address both luminal and perianal disease to achieve remission in both locations?"
    },
    {
        "id": "CD008",
        "category": "Crohn's Disease - Nutrition & Diet",
        "difficulty": "medium",
        "question": "A Crohn's patient asks about dietary approaches to achieve remission. They've heard about specific carbohydrate diet, Mediterranean diet, and exclusion diets. What evidence exists for dietary interventions in Crohn's remission, and can diet alone achieve remission or must it be combined with medical therapy?"
    },
    {
        "id": "CD009",
        "category": "Crohn's Disease - Early vs Late Disease",
        "difficulty": "hard",
        "question": "A patient with Crohn's disease diagnosed <2 years ago has moderate disease activity. Studies show early aggressive treatment improves long-term outcomes. What single treatment approach would you use in early disease to prevent complications and achieve sustained remission, compared to treatment in established disease?"
    },
    
    # ============================================================================
    # GENERAL AUTOIMMUNE - REMISSION FOCUSED
    # ============================================================================
    {
        "id": "REM001",
        "category": "Remission Strategy - Single Treatment",
        "difficulty": "very_hard",
        "question": "Current autoimmune disease treatment often requires sequential trials of 3-5 different medications before achieving remission, leading to prolonged disease activity and complications. What factors (biomarkers, disease characteristics, patient factors) should guide initial treatment selection to maximize the probability of achieving remission with a single first-line therapy?"
    },
    {
        "id": "REM002",
        "category": "Remission Strategy - Biomarkers",
        "difficulty": "very_hard",
        "question": "A patient with an autoimmune disease has elevated inflammatory markers (CRP, ESR), positive autoantibodies, and active disease. What combination of biomarkers would you measure before starting treatment to predict which single therapy is most likely to achieve remission, and how would you use these to guide treatment selection?"
    },
    {
        "id": "REM003",
        "category": "Remission Strategy - Treatment Monitoring",
        "difficulty": "hard",
        "question": "A patient starts a new treatment for autoimmune disease. Current practice often waits 3-6 months before assessing response. What early markers (clinical, laboratory, imaging) would you monitor within the first 4-8 weeks to predict if the single treatment will achieve remission, allowing early adjustment if needed?"
    },
    {
        "id": "REM004",
        "category": "Remission Strategy - Maintenance",
        "difficulty": "hard",
        "question": "A patient achieved remission on combination therapy but wants to reduce to monotherapy. What evidence supports maintaining remission with a single agent, what are the risks of relapse, and what monitoring strategy would detect early signs of disease recurrence before clinical symptoms appear?"
    },
    
    # ============================================================================
    # ORIGINAL QUESTIONS (KEPT FOR COMPARISON)
    # ============================================================================
    {
        "id": "Q001",
        "category": "Systemic Lupus Erythematosus",
        "difficulty": "medium",
        "question": "A 28-year-old woman presents with malar rash, photosensitivity, oral ulcers, and joint pain. Her ANA is positive at 1:640 with a speckled pattern, and anti-dsDNA antibodies are elevated. What is the most likely diagnosis, and what are the key diagnostic criteria you would use?"
    },
    {
        "id": "Q002",
        "category": "Sjogren's Syndrome",
        "difficulty": "hard",
        "question": "A 52-year-old woman complains of dry eyes and dry mouth for 2 years. Schirmer's test is abnormal, and she has positive anti-SSA/Ro and anti-SSB/La antibodies. What diagnostic criteria would you use to confirm Sjogren's syndrome, and what are the potential systemic complications?"
    },
    {
        "id": "Q003",
        "category": "Differential Diagnosis",
        "difficulty": "very_hard",
        "question": "A 35-year-old woman has fatigue, joint pain, positive ANA, and low complement levels. However, she also has hyperthyroidism and vitiligo. How would you differentiate between multiple autoimmune conditions versus a single systemic autoimmune disease?"
    }
]


# =============================================================================
# SCRIPT EXECUTION UTILITIES
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_section(title: str, char: str = "-", width: int = 60):
    """Print a section divider."""
    print(f"\n{title}")
    print(f"{char * width}")


def confirm_continue(message: str = "Continue?", default: bool = False) -> bool:
    """Ask user for confirmation."""
    suffix = " (Y/n): " if default else " (y/N): "
    response = input(message + suffix).strip().lower()
    
    if not response:
        return default
    return response in ('y', 'yes')


