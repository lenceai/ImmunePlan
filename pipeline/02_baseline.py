#!/usr/bin/env python3
"""
Step 02: Prompt Engineering Baseline
Book: Chapter 2 — Test base model with structured prompts before adding complexity.

Standalone: python pipeline/02_baseline.py
Output:     results/baseline_results.json
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, load_model_and_tokenizer, format_prompt, generate_response,
    save_json, print_step, RESULTS_DIR, AUTOIMMUNE_QUESTIONS, ensure_directories,
)


def run():
    print_step(2, "PROMPT ENGINEERING BASELINE")
    logger = setup_logging("02_baseline")
    ensure_directories()

    print(f"Loading model for baseline testing...")
    model, tokenizer = load_model_and_tokenizer(quantize=True)

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
                "input_tokens": tokens['input'], "output_tokens": tokens['output'],
                "word_count": len(response.split()),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"    Done in {gen_time:.1f}s ({tokens['output']} tokens)")
        except Exception as e:
            logger.error(f"Error on {q['id']}: {e}")
            results.append({"id": q['id'], "error": str(e)})

    output = RESULTS_DIR / "baseline_results.json"
    save_json(results, output)
    successful = [r for r in results if "error" not in r]
    avg_time = sum(r['time_seconds'] for r in successful) / max(len(successful), 1)
    print(f"\nBaseline: {len(successful)}/{total} successful, avg {avg_time:.1f}s")
    print(f"Saved to {output}")
    return results


def main():
    run()

if __name__ == "__main__":
    main()
