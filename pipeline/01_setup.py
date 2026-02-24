#!/usr/bin/env python3
"""
Step 01: Setup & Model Download
Book: Chapter 1 — Define reliability requirements, download model, check GPU.

Standalone: python pipeline/01_setup.py
Pipeline:   Integrated via run_pipeline.py
Output:     model_info.txt, GPU tier selection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    ensure_directories, check_cuda, get_vram_gb, select_training_tier,
    MODEL_NAME, RELIABILITY_SPEC, TRAINING_TIERS, setup_logging, print_step,
    load_model_and_tokenizer, format_prompt, generate_response, save_json,
    PROJECT_ROOT,
)


def run(skip_model_download=False):
    print_step(1, "SETUP & MODEL DOWNLOAD")
    logger = setup_logging("01_setup")
    ensure_directories()

    print("Reliability Specification:")
    print(f"  Project: {RELIABILITY_SPEC['project']}")
    print(f"  Tier: {RELIABILITY_SPEC['tier']}")
    for k, v in RELIABILITY_SPEC['quality_targets'].items():
        print(f"  {k}: {v}")

    cuda = check_cuda()
    if cuda["available"]:
        vram = cuda["total_vram_gb"]
        print(f"\nGPU: {cuda['devices'][0]['name']} ({vram:.1f} GB)")
        tier_name, tier = select_training_tier(vram)
        print(f"Training tier: {tier_name} ({tier['method']}, rank={tier['rank']})")
        print(f"  VRAM range: {tier['vram_range']}")
        logger.info(f"GPU detected: {vram:.1f} GB, tier: {tier_name}")
    else:
        print("\nNo GPU detected — CPU only mode")
        tier_name = "cpu"
        logger.info("No GPU detected")

    if skip_model_download:
        print(f"\nSkipping model download (model: {MODEL_NAME})")
        return {"tier": tier_name, "cuda": cuda}

    try:
        print(f"\nDownloading model: {MODEL_NAME}...")
        model, tokenizer = load_model_and_tokenizer(quantize=True)
        print("Model loaded successfully")

        test_q = "What are the key diagnostic criteria for systemic lupus erythematosus?"
        prompt = format_prompt(test_q, tokenizer=tokenizer)
        response, gen_time, tokens = generate_response(model, tokenizer, prompt, max_new_tokens=256)

        print(f"Inference test: {gen_time:.1f}s, {tokens['output']}/{tokens['total']} tokens")
        logger.info(f"Inference OK: {gen_time:.1f}s")

        info = {"model": MODEL_NAME, "tier": tier_name, "gpu": cuda, "test_time": gen_time}
        save_json(info, PROJECT_ROOT / "model_info.json")
        return info

    except Exception as e:
        logger.error(f"Model download failed: {e}")
        print(f"Model download failed: {e}")
        print("Pipeline can continue without GPU for RAG/agent/safety steps.")
        return {"tier": tier_name, "cuda": cuda, "error": str(e)}


def main():
    run(skip_model_download="--skip-download" in sys.argv)

if __name__ == "__main__":
    main()
