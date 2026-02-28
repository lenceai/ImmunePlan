#!/usr/bin/env python3
"""
Step 06: Test Fine-Tuned Model
Book: Chapter 5 — Compare fine-tuned model against baseline.

Standalone: conda run -n base python pipeline/06_test_model.py
Output:     results/finetuned_results.json

Improvements vs baseline:
  - Retrieves RAG context per question (same vector store as step 02)
  - Saves context_chunks for RAGAS evaluation in step 08
  - Prints side-by-side A/B quality comparison against baseline_results.json
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, save_json, load_json,
    format_prompt, generate_response, get_quantization_config,
    MODEL_NAME, DATA_DIR, MODELS_DIR, RESULTS_DIR, AUTOIMMUNE_QUESTIONS,
    AUTOIMMUNE_GROUND_TRUTH, RELIABILITY_SPEC,
)

# Same system prompt as baseline — ensures apples-to-apples A/B comparison.
_MEDICAL_SYSTEM_PROMPT = (
    "You are an autoimmune disease specialist AI assistant. "
    "Provide evidence-based, structured answers about rheumatoid arthritis, "
    "Crohn's disease, lupus (SLE), Sjogren's syndrome, and related conditions. "
    "For each response: (1) state the key diagnostic criteria or clinical facts, "
    "(2) reference relevant biomarkers and their thresholds, "
    "(3) outline evidence-based treatment options when asked, "
    "(4) state your confidence level. "
    + RELIABILITY_SPEC["medical_disclaimer"]
)


def _load_rag():
    try:
        from pipeline.lib.rag import VectorStore, RAGPipeline
        store = VectorStore()
        if store.load(str(DATA_DIR / "vector_store")):
            return RAGPipeline(store)
    except Exception:
        pass
    return None


def _print_comparison(baseline_results, finetuned_results):
    """Print side-by-side quality delta table."""
    from pipeline.lib.evaluation import ResponseQualityChecker
    checker = ResponseQualityChecker()

    b_map = {r['id']: r for r in baseline_results if 'error' not in r}
    f_map = {r['id']: r for r in finetuned_results if 'error' not in r}
    common = [q['id'] for q in AUTOIMMUNE_QUESTIONS if q['id'] in b_map and q['id'] in f_map]
    if not common:
        return

    print("\n" + "=" * 72)
    print(f"  A/B COMPARISON (Baseline vs Fine-Tuned) — {len(common)} questions")
    print("=" * 72)
    print(f"  {'ID':<8} {'Baseline':>10} {'FineTuned':>10} {'Delta':>8}  {'Words B/F':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}  {'-'*12}")

    b_scores, f_scores = [], []
    for qid in common:
        bq = checker.evaluate(b_map[qid]['response'], b_map[qid]['question'])
        fq = checker.evaluate(f_map[qid]['response'], f_map[qid]['question'])
        delta = fq.overall_score - bq.overall_score
        sign = "+" if delta >= 0 else ""
        print(f"  {qid:<8} {bq.overall_score:>10.3f} {fq.overall_score:>10.3f} "
              f"{sign}{delta:>7.3f}  {bq.word_count:>5}/{fq.word_count:<5}")
        b_scores.append(bq.overall_score)
        f_scores.append(fq.overall_score)

    avg_b = sum(b_scores) / len(b_scores)
    avg_f = sum(f_scores) / len(f_scores)
    delta = avg_f - avg_b
    sign = "+" if delta >= 0 else ""
    print(f"  {'AVERAGE':<8} {avg_b:>10.3f} {avg_f:>10.3f} {sign}{delta:>7.3f}")
    print("=" * 72)
    improvement = "IMPROVED" if delta > 0.01 else ("REGRESSION" if delta < -0.01 else "NO CHANGE")
    print(f"  Fine-tuning result: {improvement} ({sign}{delta:.3f} overall quality)")


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

    rag = _load_rag()
    if rag:
        print(f"Vector store loaded ({len(rag.vector_store.chunks)} chunks)")
    else:
        print("Vector store not available — run step 04 for RAGAS evaluation")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=get_quantization_config(),
        torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    print("Model loaded")

    results = []
    total = len(AUTOIMMUNE_QUESTIONS)
    for i, q in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"  [{i}/{total}] {q['id']}: {q['category']}")
        try:
            # Fine-tuned model trained with /no_think — use same mode for inference.
            # /think would conflict with the LoRA's learned direct-response behaviour.
            prompt = format_prompt(q['question'], system_prompt=_MEDICAL_SYSTEM_PROMPT,
                                   tokenizer=tokenizer, use_thinking=False)
            response, gen_time, tokens = generate_response(
                model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3,
            )

            context_chunks = []
            retrieval_quality = None
            if rag:
                try:
                    retrieval = rag.retrieve(q['question'], top_k=5)
                    context_chunks = [c.text for c in retrieval.chunks]
                    retrieval_quality = round(retrieval.quality_score, 3)
                except Exception as e:
                    logger.warning(f"RAG retrieval failed for {q['id']}: {e}")

            results.append({
                "id": q['id'],
                "category": q['category'],
                "difficulty": q['difficulty'],
                "question": q['question'],
                "response": response,
                "context_chunks": context_chunks,
                "retrieval_quality": retrieval_quality,
                "reference": AUTOIMMUNE_GROUND_TRUTH.get(q['id'], ""),
                "time_seconds": round(gen_time, 2),
                "input_tokens": tokens['input'],
                "output_tokens": tokens['output'],
                "word_count": len(response.split()),
                "model_type": info.get('method', 'finetuned'),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"    Done in {gen_time:.1f}s ({tokens['output']} tokens)")
        except Exception as e:
            logger.error(f"Error on {q['id']}: {e}")
            results.append({"id": q['id'], "error": str(e)})

    output = RESULTS_DIR / "finetuned_results.json"
    save_json(results, output)
    successful = [r for r in results if "error" not in r]
    print(f"\nFine-tuned: {len(successful)}/{total} successful")
    print(f"Saved to {output}")

    # A/B comparison against baseline
    baseline = load_json(RESULTS_DIR / "baseline_results.json")
    if baseline:
        _print_comparison(baseline, results)
    else:
        print("(No baseline_results.json found — run step 02 for A/B comparison)")

    return results


def main():
    run()

if __name__ == "__main__":
    main()
