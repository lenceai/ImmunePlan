#!/usr/bin/env python3
"""
Step 04: Build RAG Pipeline
Book: Chapters 3-4 — Vector store, embeddings, hybrid retrieval for grounding.

Standalone: conda run -n base python pipeline/04_build_rag.py
Output:     data/vector_store/   (chunks.json + embeddings.npy)
            results/rag_quality.json

Measures retrieval quality with RAGAS context_precision metric
on the AUTOIMMUNE_QUESTIONS benchmark queries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, DATA_DIR, RESULTS_DIR, ensure_directories,
    AUTOIMMUNE_QUESTIONS, save_json,
)
from pipeline.lib.rag import VectorStore, RAGPipeline
from pipeline.lib.evaluation import RAGASEvaluator


def _retrieval_quality_report(pipeline: RAGPipeline, evaluator: RAGASEvaluator) -> list:
    """
    Run every benchmark question through retrieval and score context precision.
    Returns a list of per-query quality dicts.
    """
    report = []
    for q in AUTOIMMUNE_QUESTIONS:
        try:
            result = pipeline.retrieve(q['question'], top_k=5)
            chunk_texts = [c.text for c in result.chunks]
            cp = evaluator.context_precision(q['question'], chunk_texts)
            report.append({
                "id": q['id'],
                "category": q['category'],
                "difficulty": q['difficulty'],
                "chunks_retrieved": len(result.chunks),
                "retrieval_quality_score": round(result.quality_score, 3),
                "context_precision": round(cp, 3),
                "sufficient_context": result.sufficient_context,
                "retrieval_method": result.retrieval_method,
            })
        except Exception as e:
            report.append({"id": q['id'], "error": str(e)})
    return report


def run():
    print_step(4, "BUILD RAG PIPELINE")
    logger = setup_logging("04_build_rag")
    ensure_directories()

    chunks_file = DATA_DIR / "paper_chunks.json"
    if not chunks_file.exists():
        print(f"No chunks file at {chunks_file}")
        print("Run step 03 first, or RAG will use keyword-only fallback.")
        return {"status": "no_data"}

    store = VectorStore()
    pipeline = RAGPipeline(store)

    print(f"Ingesting chunks from {chunks_file}...")
    pipeline.ingest_training_data(str(chunks_file))
    print(f"  Chunks embedded : {len(store.chunks)}")
    print(f"  Embedding dim   : {store.dimension}")

    # --- Retrieval quality using RAGAS context_precision ---
    print("\nEvaluating retrieval quality (RAGAS context_precision)...")
    evaluator = RAGASEvaluator()
    quality_report = _retrieval_quality_report(pipeline, evaluator)

    valid = [r for r in quality_report if 'error' not in r]
    if valid:
        avg_cp = sum(r['context_precision'] for r in valid) / len(valid)
        avg_rq = sum(r['retrieval_quality_score'] for r in valid) / len(valid)
        sufficient = sum(1 for r in valid if r['sufficient_context'])

        print(f"\n  {'ID':<8} {'CP':>6} {'RQ':>6} {'Chunks':>7} {'Sufficient':>10}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
        for r in valid:
            ok = "YES" if r['sufficient_context'] else "NO"
            print(f"  {r['id']:<8} {r['context_precision']:>6.3f} {r['retrieval_quality_score']:>6.3f} "
                  f"{r['chunks_retrieved']:>7} {ok:>10}")
        print(f"\n  Average context_precision : {avg_cp:.3f}")
        print(f"  Average retrieval_quality : {avg_rq:.3f}")
        print(f"  Sufficient context        : {sufficient}/{len(valid)}")

        # Quality threshold check against RELIABILITY_SPEC
        precision_target = 0.7   # want 70%+ of retrieved chunks to be relevant
        if avg_cp >= precision_target:
            print(f"\n  ✓ Context precision ({avg_cp:.3f}) meets target ({precision_target})")
        else:
            print(f"\n  ✗ Context precision ({avg_cp:.3f}) BELOW target ({precision_target}) "
                  f"— consider re-running step 03 with more data")

        save_json(
            {
                "avg_context_precision": round(avg_cp, 3),
                "avg_retrieval_quality": round(avg_rq, 3),
                "sufficient_context_rate": round(sufficient / len(valid), 3),
                "per_query": quality_report,
            },
            RESULTS_DIR / "rag_quality.json",
        )
        print(f"\n  RAG quality report saved to {RESULTS_DIR / 'rag_quality.json'}")
    else:
        print("  No valid retrieval results — check embeddings")

    logger.info(
        f"RAG built: {len(store.chunks)} chunks, "
        f"avg context_precision={avg_cp:.3f}" if valid else "no quality data"
    )
    return {
        "chunks": len(store.chunks),
        "embeddings": len(store.embeddings),
        "avg_context_precision": round(avg_cp, 3) if valid else None,
    }


def main():
    run()

if __name__ == "__main__":
    main()
