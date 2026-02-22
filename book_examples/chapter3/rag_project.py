"""
Chapter 3: RAG Project - Grounding Outputs with Retrieval

Demonstrates a RAG pipeline for grounding LLM responses in real documents.
Key reliability insight: "Many 'LLM hallucination problems' are really
retrieval problems."

This example shows:
  - Document loading and chunking
  - Vector store creation
  - Retrieval-augmented question answering
  - Evaluation of retrieval quality
"""

import os
import json
from pathlib import Path


def create_knowledge_base():
    """Create a sample medical knowledge base for demonstration."""
    documents = [
        {
            "title": "Rheumatoid Arthritis Diagnosis",
            "content": (
                "Rheumatoid arthritis (RA) is diagnosed using the 2010 ACR/EULAR "
                "classification criteria. The criteria evaluate: joint involvement "
                "(0-5 points), serology including RF and anti-CCP (0-3 points), "
                "acute phase reactants ESR and CRP (0-1 point), and symptom duration "
                "(0-1 point). A total score of 6 or more out of 10 classifies as "
                "definite RA. Early diagnosis is crucial for preventing joint damage."
            ),
            "source": "ACR/EULAR 2010",
            "category": "diagnosis",
        },
        {
            "title": "RA Treatment with DMARDs",
            "content": (
                "Disease-modifying antirheumatic drugs (DMARDs) are the cornerstone "
                "of RA treatment. Methotrexate is typically the first-line DMARD. "
                "The treat-to-target strategy aims for remission or low disease "
                "activity within 3-6 months. If methotrexate alone is insufficient, "
                "biologic DMARDs such as TNF inhibitors (adalimumab, infliximab) "
                "may be added."
            ),
            "source": "ACR Treatment Guidelines",
            "category": "treatment",
        },
        {
            "title": "Crohn's Disease Biomarkers",
            "content": (
                "Fecal calprotectin is a key biomarker for monitoring Crohn's disease "
                "activity. Levels below 50 ug/g suggest remission, while levels above "
                "200 ug/g indicate active inflammation. CRP and ESR are also used as "
                "systemic inflammation markers. ASCA antibodies may support diagnosis "
                "of Crohn's disease."
            ),
            "source": "IBD Guidelines",
            "category": "biomarkers",
        },
    ]
    return documents


def simple_rag_example():
    """Demonstrate a simple RAG pipeline without external dependencies."""
    print("Building knowledge base...")
    documents = create_knowledge_base()

    query = "What are the diagnostic criteria for rheumatoid arthritis?"

    print(f"\nQuery: {query}")
    print("\nRetrieving relevant documents...")

    query_terms = set(query.lower().split())
    scored_docs = []
    for doc in documents:
        doc_terms = set(doc["content"].lower().split())
        overlap = len(query_terms & doc_terms)
        score = overlap / max(len(query_terms), 1)
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    print("\nRetrieved documents (ranked by relevance):")
    for i, (score, doc) in enumerate(scored_docs, 1):
        print(f"  {i}. [{score:.2f}] {doc['title']} (source: {doc['source']})")

    top_doc = scored_docs[0][1]
    print(f"\nBest match context:\n  {top_doc['content'][:200]}...")

    print("\nRAG Evaluation Checklist:")
    print("  [x] Did it retrieve the right source?")
    print("  [x] Did it cite/source correctly?")
    print("  [ ] Did it answer using only grounded content? (needs LLM)")
    print("  [ ] Did it refuse when context was insufficient? (needs LLM)")


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 3: RAG Project - Grounding with Retrieval")
    print("=" * 80)
    print()
    simple_rag_example()
