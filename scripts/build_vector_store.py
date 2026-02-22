#!/usr/bin/env python3
"""
Build Vector Store from Training Data

Ingests the chunked paper data (from script 3) into the vector store
used by the RAG pipeline. This is a prerequisite for grounded responses.

Usage:
    python scripts/build_vector_store.py

Input:
    - data/paper_chunks.json: Chunked papers from script 3

Output:
    - data/vector_store/: Persisted vector store (chunks.json + embeddings.npy)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reliability.config import RAG_CONFIG
from reliability.rag import VectorStore, RAGPipeline


def main():
    print("=" * 60)
    print("BUILD VECTOR STORE FROM TRAINING DATA")
    print("=" * 60)

    data_dir = Path("./data")
    chunks_file = data_dir / "paper_chunks.json"

    if not chunks_file.exists():
        print(f"\nNo chunks file found at {chunks_file}")
        print("Run script 3 (download_papers.py) first to create training data.")
        print("\nAlternatively, the RAG pipeline will use keyword-based fallback.")
        sys.exit(0)

    store = VectorStore()
    pipeline = RAGPipeline(store)

    print(f"\nIngesting chunks from {chunks_file}...")
    pipeline.ingest_training_data(str(chunks_file))

    print(f"\nVector store built:")
    print(f"  Chunks: {len(store.chunks)}")
    print(f"  Embeddings: {len(store.embeddings)}")
    print(f"  Stored at: {RAG_CONFIG.vector_db_path}")

    print("\nTesting retrieval...")
    test_queries = [
        "diagnostic criteria for rheumatoid arthritis",
        "Crohn's disease treatment with biologics",
        "autoimmune disease biomarkers",
    ]

    for query in test_queries:
        result = pipeline.retrieve(query)
        print(f"\n  Query: {query}")
        print(f"  Chunks found: {len(result.chunks)}")
        print(f"  Quality: {result.quality_score:.2f}")
        print(f"  Sufficient context: {result.sufficient_context}")
        if result.chunks:
            print(f"  Top chunk: {result.chunks[0].paper_title} ({result.chunks[0].section})")

    print("\n" + "=" * 60)
    print("Vector store built successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
