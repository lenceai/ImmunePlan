#!/usr/bin/env python3
"""
Step 06: Test Fine-Tuned Model
Book: Chapter 5 — Compare fine-tuned model against baseline.

Standalone: python pipeline/06_test_model.py
Output:     results/finetuned_results.json
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, save_json, load_json,
    format_prompt, generate_response, get_quantization_config,
    MODEL_NAME, MODELS_DIR, RESULTS_DIR, AUTOIMMUNE_QUESTIONS,
)


def run():
    print_step(6, "TEST FINE-TUNED MODEL")
    logger = setup_logging("06_test_model")
    ensure_directories()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_path = MODELS_DIR / "finetuned_model"
    if not model_path.exists():
        print(f"No fine-tuned model at {model_path}")
        print("Run step 05 first, or skip to step 07.")
        return {"status": "skipped"}

    info = load_json(model_path / "training_info.json") or {}
    print(f"Loading fine-tuned model ({info.get('method', 'unknown')}, rank={info.get('rank', '?')})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=get_quantization_config(),
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    print("Model loaded")

    results = []
    total = len(AUTOIMMUNE_QUESTIONS)
    for i, q in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"  [{i}/{total}] {q['id']}: {q['category']}")
        try:
            prompt = format_prompt(q['question'], tokenizer=tokenizer)
            response, gen_time, tokens = generate_response(
                model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7,
            )
            results.append({
                "id": q['id'], "category": q['category'], "difficulty": q['difficulty'],
                "question": q['question'], "response": response,
                "time_seconds": round(gen_time, 2),
                "output_tokens": tokens['output'], "word_count": len(response.split()),
                "model_type": info.get('method', 'finetuned'),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"    Done in {gen_time:.1f}s")
        except Exception as e:
            logger.error(f"Error on {q['id']}: {e}")
            results.append({"id": q['id'], "error": str(e)})

    output = RESULTS_DIR / "finetuned_results.json"
    save_json(results, output)
    successful = [r for r in results if "error" not in r]
    print(f"\nFine-tuned: {len(successful)}/{total} successful")
    print(f"Saved to {output}")
    return results


def main():
    run()

if __name__ == "__main__":
    main()
