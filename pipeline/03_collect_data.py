#!/usr/bin/env python3
"""
Step 03: Collect Training Data
Book: Chapters 3-4 prep — Download research papers, extract text, chunk for RAG & fine-tuning.

Standalone: python pipeline/03_collect_data.py
Output:     data/raw_papers.json, data/paper_chunks.json, data/papers_training_data.json
"""
import sys
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, save_json, load_json, print_step, DATA_DIR, PUBMED_EMAIL,
    MODEL_NAME, ensure_directories, format_prompt,
)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 500

ARXIV_QUERIES = [
    "rheumatoid arthritis diagnosis treatment", "rheumatoid arthritis pathogenesis",
    "rheumatoid arthritis biomarkers", "rheumatoid arthritis DMARDs biologics",
    "rheumatoid arthritis remission", "Crohn's disease inflammatory bowel",
    "Crohn disease diagnosis treatment", "Crohn's disease biomarkers",
    "Crohn disease biologics", "Crohn's disease remission",
    "autoimmune disease diagnosis treatment", "autoimmune disease machine learning",
    "immune system dysfunction", "vagus nerve immune system",
    "fasting immune system", "diet autoimmune disease",
]

PUBMED_QUERIES = [
    "rheumatoid arthritis[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "Crohn disease[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "autoimmune disease[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
]


def chunk_text(text: str) -> List[Dict]:
    paragraphs = text.split('\n\n')
    chunks, current, start = [], "", 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > CHUNK_SIZE:
            if len(current) >= MIN_CHUNK_SIZE:
                chunks.append({"text": current.strip(), "char_start": start})
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else ""
            current = overlap + "\n\n" + para if overlap else para
            start += len(current)
        else:
            current = current + "\n\n" + para if current else para
    if len(current) >= MIN_CHUNK_SIZE:
        chunks.append({"text": current.strip(), "char_start": start})
    return chunks


def identify_section(text: str) -> str:
    t = text.lower()[:500]
    if any(w in t for w in ['abstract', 'summary']):
        return "abstract"
    if any(w in t for w in ['introduction', 'background']):
        return "introduction"
    if any(w in t for w in ['method', 'materials']):
        return "methods"
    if any(w in t for w in ['result', 'finding']):
        return "results"
    if any(w in t for w in ['discussion', 'conclusion']):
        return "discussion"
    if any(w in t for w in ['reference', 'bibliography']):
        return "references"
    return "body"


def search_arxiv(queries, max_per_query=20, logger=None):
    try:
        import arxiv
    except ImportError:
        if logger:
            logger.warning("arxiv package not installed, skipping arXiv search")
        return []

    papers = []
    client = arxiv.Client()
    for q in queries:
        try:
            search = arxiv.Search(query=q, max_results=max_per_query, sort_by=arxiv.SortCriterion.SubmittedDate)
            for r in client.results(search):
                if r.summary and len(r.summary) >= 100:
                    aid = r.entry_id.split('/')[-1]
                    papers.append({
                        "source": "arXiv", "arxiv_id": aid, "title": r.title,
                        "abstract": r.summary, "pub_date": r.published.strftime("%Y-%m-%d") if r.published else "",
                        "authors": [a.name for a in r.authors[:5]],
                        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                    })
            time.sleep(0.5)
        except Exception as e:
            if logger:
                logger.error(f"arXiv error: {e}")
    return papers


def search_pubmed(queries, max_per_query=10, logger=None):
    if not PUBMED_EMAIL:
        if logger:
            logger.warning("PUBMED_EMAIL not set, skipping PubMed")
        return []

    try:
        from Bio import Entrez
    except ImportError:
        if logger:
            logger.warning("biopython not installed, skipping PubMed")
        return []

    Entrez.email = PUBMED_EMAIL
    papers = []
    for q in queries:
        try:
            handle = Entrez.esearch(db="pubmed", term=q, retmax=max_per_query)
            record = Entrez.read(handle)
            handle.close()
            pmids = record.get("IdList", [])
            if pmids:
                handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                for rec in records.get("PubmedArticle", []):
                    article = rec.get("MedlineCitation", {}).get("Article", {})
                    abstract_list = article.get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join(str(a) for a in abstract_list) if isinstance(abstract_list, list) else str(abstract_list)
                    if abstract and len(abstract) >= 100:
                        papers.append({
                            "source": "PubMed", "title": article.get("ArticleTitle", ""),
                            "abstract": abstract,
                        })
            time.sleep(0.5)
        except Exception as e:
            if logger:
                logger.error(f"PubMed error: {e}")
    return papers


def deduplicate(papers):
    seen = set()
    unique = []
    for p in papers:
        key = ''.join(c for c in p['title'].lower() if c.isalnum())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def run():
    print_step(3, "COLLECT TRAINING DATA")
    logger = setup_logging("03_collect_data")
    ensure_directories()

    papers = []
    print("Searching arXiv...")
    papers.extend(search_arxiv(ARXIV_QUERIES, logger=logger))
    print(f"  Found {len(papers)} arXiv papers")

    print("Searching PubMed...")
    pubmed = search_pubmed(PUBMED_QUERIES, logger=logger)
    papers.extend(pubmed)
    print(f"  Found {len(pubmed)} PubMed papers")

    papers = deduplicate(papers)
    print(f"  After dedup: {len(papers)} unique papers")

    all_chunks = []
    for p in papers:
        text = p.get("abstract", "")
        if not text:
            continue
        for j, chunk in enumerate(chunk_text(text)):
            section = identify_section(chunk['text'])
            if section == "references":
                continue
            all_chunks.append({
                "paper_title": p['title'], "source": p['source'],
                "section": section, "text": chunk['text'], "chunk_index": j,
            })

    save_json([{k: v for k, v in p.items() if k != 'full_text'} for p in papers],
              DATA_DIR / "raw_papers.json")
    save_json(all_chunks, DATA_DIR / "paper_chunks.json")

    training_data = []
    for chunk in all_chunks:
        instruction = f"Analyze this research excerpt about autoimmune disease:\n\nTitle: {chunk['paper_title']}\n\n{chunk['text']}"
        training_data.append({
            "text": format_prompt(instruction),
            "paper_title": chunk['paper_title'], "source": chunk['source'],
            "section": chunk['section'],
        })
    save_json(training_data, DATA_DIR / "papers_training_data.json")

    print(f"\nCollected: {len(papers)} papers, {len(all_chunks)} chunks, {len(training_data)} training examples")
    return {"papers": len(papers), "chunks": len(all_chunks), "training": len(training_data)}


def main():
    run()

if __name__ == "__main__":
    main()
