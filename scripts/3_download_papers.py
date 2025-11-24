#!/usr/bin/env python3
"""
Script 3: Download Research Papers
Version: 1.0.0

Purpose: Download latest research papers from PubMed and arXiv for training.

Usage:
    python scripts/3_download_papers.py

Requirements:
    - PUBMED_EMAIL must be set in .env file
    - Internet connection for API access

Output:
    - data/raw_papers.json: Raw paper data from APIs
    - data/papers_training_data.json: Formatted training data
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import requests
from Bio import Entrez
import arxiv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Ensure data directory exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# PubMed search queries
PUBMED_QUERIES = [
    "systemic lupus erythematosus[Title] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "rheumatoid arthritis[Title] AND diagnosis[Title] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "autoimmune disease[Title] AND machine learning[Title/Abstract]",
    "Sjogren syndrome[Title] AND (2024[PDAT] OR 2025[PDAT])",
    "systemic sclerosis[Title] AND (2024[PDAT] OR 2025[PDAT])",
    "polymyositis[Title] AND diagnosis[Title]",
    "vasculitis[Title] AND (2024[PDAT] OR 2025[PDAT])",
    "Behçet disease[Title] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
]

# arXiv search queries
ARXIV_QUERIES = [
    "autoimmune disease diagnosis machine learning",
    "lupus artificial intelligence",
    "rheumatoid arthritis deep learning",
    "autoimmune disease classification",
]


def search_pubmed(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search PubMed for papers.
    
    Args:
        query: PubMed search query
        max_results: Maximum number of results
    
    Returns:
        List of paper dictionaries
    """
    if not PUBMED_EMAIL:
        print("WARNING: PUBMED_EMAIL not set. Skipping PubMed search.")
        return []
    
    Entrez.email = PUBMED_EMAIL
    
    papers = []
    try:
        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        
        if not pmids:
            return papers
        
        # Fetch details
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        for record in records.get("PubmedArticle", []):
            try:
                medline = record.get("MedlineCitation", {})
                article = medline.get("Article", {})
                
                # Extract title
                title = article.get("ArticleTitle", "")
                
                # Extract abstract
                abstract_text = ""
                abstract_list = article.get("Abstract", {}).get("AbstractText", [])
                if isinstance(abstract_list, list):
                    abstract_text = " ".join([str(a) for a in abstract_list])
                elif isinstance(abstract_list, str):
                    abstract_text = abstract_list
                
                # Skip if no abstract
                if not abstract_text or len(abstract_text) < 100:
                    continue
                
                # Extract journal
                journal = article.get("Journal", {}).get("Title", "")
                
                # Extract publication date
                pub_date = ""
                pub_date_node = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                if pub_date_node:
                    year = pub_date_node.get("Year", "")
                    month = pub_date_node.get("Month", "")
                    pub_date = f"{year}-{month}" if year else ""
                
                # Extract authors (first 5)
                authors = []
                author_list = article.get("AuthorList", [])
                for author in author_list[:5]:
                    last_name = author.get("LastName", "")
                    first_name = author.get("ForeName", "")
                    if last_name:
                        authors.append(f"{first_name} {last_name}".strip())
                
                # Extract PMID
                pmid = medline.get("PMID", "")
                
                paper = {
                    "source": "PubMed",
                    "pmid": str(pmid),
                    "arxiv_id": "",
                    "title": title,
                    "abstract": abstract_text,
                    "journal": journal,
                    "pub_date": pub_date,
                    "authors": authors
                }
                
                papers.append(paper)
                
            except Exception as e:
                print(f"  Error processing PubMed record: {str(e)}")
                continue
        
        # Rate limiting
        time.sleep(1)
        
    except Exception as e:
        print(f"Error searching PubMed: {str(e)}")
    
    return papers


def search_arxiv(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search arXiv for papers.
    
    Args:
        query: arXiv search query
        max_results: Maximum number of results
    
    Returns:
        List of paper dictionaries
    """
    papers = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in search.results():
            # Skip if no abstract or too short
            if not result.summary or len(result.summary) < 100:
                continue
            
            paper = {
                "source": "arXiv",
                "pmid": "",
                "arxiv_id": result.entry_id.split('/')[-1],
                "title": result.title,
                "abstract": result.summary,
                "journal": "arXiv",
                "pub_date": result.published.strftime("%Y-%m-%d") if result.published else "",
                "authors": [author.name for author in result.authors[:5]]
            }
            
            papers.append(paper)
        
        # Rate limiting
        time.sleep(1)
        
    except Exception as e:
        print(f"Error searching arXiv: {str(e)}")
    
    return papers


def remove_duplicates(papers: List[Dict]) -> List[Dict]:
    """
    Remove duplicate papers by title similarity.
    
    Args:
        papers: List of paper dictionaries
    
    Returns:
        Deduplicated list
    """
    seen_titles = set()
    unique_papers = []
    
    for paper in papers:
        title_lower = paper['title'].lower().strip()
        # Simple deduplication by exact title match
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(paper)
    
    return unique_papers


def download_papers() -> List[Dict]:
    """
    Download papers from all sources.
    
    Returns:
        List of all papers
    """
    all_papers = []
    
    print(f"\n{'='*60}")
    print("DOWNLOADING RESEARCH PAPERS")
    print(f"{'='*60}\n")
    
    # PubMed papers
    print("Searching PubMed...")
    for i, query in enumerate(PUBMED_QUERIES, 1):
        print(f"  [{i}/{len(PUBMED_QUERIES)}] Query: {query[:60]}...")
        papers = search_pubmed(query, max_results=15)
        all_papers.extend(papers)
        print(f"    Found {len(papers)} papers")
    
    # arXiv papers
    print(f"\nSearching arXiv...")
    for i, query in enumerate(ARXIV_QUERIES, 1):
        print(f"  [{i}/{len(ARXIV_QUERIES)}] Query: {query}")
        papers = search_arxiv(query, max_results=10)
        all_papers.extend(papers)
        print(f"    Found {len(papers)} papers")
    
    # Remove duplicates
    print(f"\nRemoving duplicates...")
    print(f"  Before: {len(all_papers)} papers")
    all_papers = remove_duplicates(all_papers)
    print(f"  After: {len(all_papers)} papers")
    
    return all_papers


def process_papers_for_training(papers: List[Dict]) -> List[Dict]:
    """
    Process papers into training format.
    
    Args:
        papers: Raw paper data
    
    Returns:
        Formatted training data
    """
    training_data = []
    
    for paper in papers:
        # Format as instruction-following data
        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        
        training_text = f"""<｜begin▁of▁sentence｜>User: Summarize the following research paper and explain its relevance to autoimmune disease diagnosis:

{paper_text}RetryRMfinish"""
        
        training_data.append({
            "text": training_text,
            "source": paper['source'],
            "paper_id": paper.get('pmid') or paper.get('arxiv_id', ''),
            "title": paper['title']
        })
    
    return training_data


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 3: DOWNLOAD RESEARCH PAPERS")
    print("="*60 + "\n")
    
    try:
        # Download papers
        papers = download_papers()
        
        if not papers:
            print("WARNING: No papers downloaded. Check your PUBMED_EMAIL setting.")
            sys.exit(1)
        
        # Save raw papers
        raw_file = os.path.join(DATA_DIR, "raw_papers.json")
        with open(raw_file, 'w') as f:
            json.dump(papers, f, indent=2)
        print(f"\n✓ Raw papers saved to {raw_file}")
        
        # Process for training
        training_data = process_papers_for_training(papers)
        
        # Save training data
        training_file = os.path.join(DATA_DIR, "papers_training_data.json")
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"✓ Training data saved to {training_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}\n")
        print(f"Total Papers: {len(papers)}")
        
        source_counts = {}
        for paper in papers:
            source = paper['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        print(f"\nTraining Examples: {len(training_data)}")
        
        print("\n" + "="*60)
        print("✓ Script 3 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

