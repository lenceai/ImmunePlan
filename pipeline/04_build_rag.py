#!/usr/bin/env python3
"""
Step 04: Build RAG Pipeline
Book: Chapters 3-4 — Vector store, embeddings, hybrid retrieval for grounding.

Standalone: python pipeline/04_build_rag.py
Output:     data/vector_store/ (chunks.json + embeddings.npy)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import setup_logging, print_step, DATA_DIR, ensure_directories
from pipeline.lib.rag import VectorStore, RAGPipeline
from pipeline.lib.prompts import rephrase_query_for_retrieval


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
    print(f"  Chunks: {len(store.chunks)}")
    print(f"  Embeddings: {len(store.embeddings)}")

    print("\nTesting retrieval...")
    test_queries = [
        "diagnostic criteria for rheumatoid arthritis",
        "Crohn's disease treatment with biologics",
        "autoimmune disease biomarkers",
    ]

    for query in test_queries:
        variations = rephrase_query_for_retrieval(query)
        result = pipeline.retrieve(query)
        print(f"\n  Query: {query}")
        print(f"  Variations: {len(variations)}")
        print(f"  Chunks found: {len(result.chunks)}")
        print(f"  Quality: {result.quality_score:.2f}")
        print(f"  Sufficient context: {result.sufficient_context}")

    print(f"\nVector store saved to {DATA_DIR / 'vector_store'}")
    logger.info(f"RAG built: {len(store.chunks)} chunks, {len(store.embeddings)} embeddings")
    return {"chunks": len(store.chunks), "embeddings": len(store.embeddings)}


def main():
    run()

if __name__ == "__main__":
    main()
