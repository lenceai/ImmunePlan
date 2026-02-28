#!/usr/bin/env python3
"""
Step 02: Prompt Engineering Baseline
Book: Chapter 2 — Test base model with structured prompts before adding complexity.

Standalone: conda run -n base python pipeline/02_baseline.py
Output:     results/baseline_results.json

For each question:
  - Generates a pure-LLM response (no RAG context fed in)
  - Retrieves context from vector store (if available) for post-hoc evaluation
  - Saves both response and context so step 08 can compute RAGAS metrics
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, load_model_and_tokenizer, format_prompt, generate_response,
    save_json, print_step, DATA_DIR, RESULTS_DIR, AUTOIMMUNE_QUESTIONS,
    AUTOIMMUNE_GROUND_TRUTH, ensure_directories, RELIABILITY_SPEC,
)

# System prompt used for both baseline and fine-tuned testing.
# Guides the model to produce structured, evidence-based medical answers.
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
    """Try to load the vector store for post-hoc context retrieval."""
    try:
        from pipeline.lib.rag import VectorStore, RAGPipeline
        store = VectorStore()
        if store.load(str(DATA_DIR / "vector_store")):
            return RAGPipeline(store)
    except Exception:
        pass
    return None


def run():
    print_step(2, "PROMPT ENGINEERING BASELINE")
    logger = setup_logging("02_baseline")
    ensure_directories()

    rag = _load_rag()
    if rag:
        print(f"Vector store loaded ({len(rag.vector_store.chunks)} chunks) — context will be saved for RAGAS evaluation")
    else:
        print("Vector store not available — run step 04 first for RAGAS evaluation. Continuing baseline only.")

    print(f"Loading model for baseline testing...")
    model, tokenizer = load_model_and_tokenizer(quantize=True)

    results = []
    total = len(AUTOIMMUNE_QUESTIONS)

    for i, q in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"  [{i}/{total}] {q['id']}: {q['category']}")
        try:
            prompt = format_prompt(q['question'], system_prompt=_MEDICAL_SYSTEM_PROMPT,
                                   tokenizer=tokenizer)
            response, gen_time, tokens = generate_response(
                model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3,
            )

            # Retrieve context for post-hoc groundedness evaluation
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
                "context_chunks": context_chunks,          # for RAGAS faithfulness/recall
                "retrieval_quality": retrieval_quality,    # RAG retrieval score
                "reference": AUTOIMMUNE_GROUND_TRUTH.get(q['id'], ""),  # for RAGAS correctness
                "time_seconds": round(gen_time, 2),
                "input_tokens": tokens['input'],
                "output_tokens": tokens['output'],
                "word_count": len(response.split()),
                "model_type": "baseline",
                "timestamp": datetime.now().isoformat(),
            })
            print(f"    Done in {gen_time:.1f}s ({tokens['output']} tokens, {len(context_chunks)} context chunks)")
        except Exception as e:
            logger.error(f"Error on {q['id']}: {e}")
            results.append({"id": q['id'], "error": str(e)})

    output = RESULTS_DIR / "baseline_results.json"
    save_json(results, output)
    successful = [r for r in results if "error" not in r]
    avg_time = sum(r['time_seconds'] for r in successful) / max(len(successful), 1)
    with_context = sum(1 for r in successful if r.get('context_chunks'))
    print(f"\nBaseline: {len(successful)}/{total} successful, avg {avg_time:.1f}s")
    print(f"  {with_context}/{len(successful)} responses have retrieved context for RAGAS evaluation")
    print(f"Saved to {output}")
    return results


def main():
    run()

if __name__ == "__main__":
    main()
