#!/usr/bin/env python3
"""
Script 3: Download and Process Research Papers (Full PDF Version)
Version: 2.0.0

Purpose: Download full PDFs from PubMed Central and arXiv, extract text,
         chunk it intelligently, and create training data for fine-tuning.

Usage:
    python scripts/3_download_papers.py

Requirements:
    - PUBMED_EMAIL must be set in .env file
    - Internet connection for API access
    - PyMuPDF (fitz) for PDF extraction

Output:
    - data/pdfs/: Downloaded PDF files
    - data/raw_papers.json: Raw paper metadata
    - data/paper_chunks.json: Chunked text data
    - data/papers_training_data.json: Formatted training data
"""

import sys
import os
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    save_json,
    load_json,
    print_header,
    print_section,
    format_prompt,
)

# Third-party imports
from Bio import Entrez
import arxiv

# PDF extraction - try to import, provide helpful error if missing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠ PyMuPDF not installed. Run: pip install pymupdf")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunk settings for training data
CHUNK_SIZE = 1500  # Characters per chunk (roughly 300-400 tokens)
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity
MIN_CHUNK_SIZE = 500  # Minimum chunk size to keep

# Download settings
MAX_PAPERS_PER_QUERY_PUBMED = 15  # Increased for RA and Crohn's focus
MAX_PAPERS_PER_QUERY_ARXIV = 35  # Increased for RA and Crohn's focus (all have PDFs - PRIMARY SOURCE)
PDF_DOWNLOAD_TIMEOUT = 60  # seconds
MAX_CONCURRENT_DOWNLOADS = 3

# Search queries - Focus on open access and immune-related research
# PubMed queries filtered for open access when possible
PUBMED_QUERIES = [
    # ===== PRIMARY FOCUS: RHEUMATOID ARTHRITIS (RA) =====
    "rheumatoid arthritis[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "rheumatoid arthritis[Title] AND diagnosis[Title] AND open access[Filter]",
    "rheumatoid arthritis[Title] AND treatment[Title] AND open access[Filter]",
    "rheumatoid arthritis[Title] AND biomarkers[Title/Abstract] AND open access[Filter]",
    "rheumatoid arthritis[Title] AND DMARDs[Title/Abstract] AND open access[Filter]",
    "rheumatoid arthritis[Title] AND biologics[Title/Abstract] AND open access[Filter]",
    
    # ===== PRIMARY FOCUS: CROHN'S DISEASE =====
    "Crohn disease[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "Crohn's disease[Title] AND diagnosis[Title] AND open access[Filter]",
    "Crohn disease[Title] AND treatment[Title] AND open access[Filter]",
    "Crohn's disease[Title] AND biomarkers[Title/Abstract] AND open access[Filter]",
    "Crohn disease[Title] AND biologics[Title/Abstract] AND open access[Filter]",
    "inflammatory bowel disease[Title] AND Crohn[Title/Abstract] AND open access[Filter]",
    
    # General autoimmune and immune system
    "autoimmune disease[Title] AND open access[Filter] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "immune system[Title] AND open access[Filter]",
    "immunotherapy[Title] AND autoimmune[Title] AND open access[Filter]",
]

# arXiv queries - PRIMARY SOURCE (all have free PDFs)
# Focus on immune system, autoimmune, treatment, and related topics
ARXIV_QUERIES = [
    # ===== PRIMARY FOCUS: RHEUMATOID ARTHRITIS (RA) =====
    "rheumatoid arthritis diagnosis treatment",
    "rheumatoid arthritis pathogenesis",
    "rheumatoid arthritis machine learning",
    "RA rheumatoid arthritis biomarkers",
    "rheumatoid arthritis DMARDs biologics",
    "rheumatoid arthritis inflammation",
    "rheumatoid arthritis joint damage",
    "rheumatoid arthritis early diagnosis",
    "rheumatoid arthritis treatment response",
    "rheumatoid arthritis remission",
    "rheumatoid arthritis genetics",
    "rheumatoid arthritis cytokines",
    "rheumatoid arthritis autoantibodies",
    "rheumatoid arthritis imaging",
    "rheumatoid arthritis prediction",
    
    # ===== PRIMARY FOCUS: CROHN'S DISEASE =====
    "Crohn's disease inflammatory bowel",
    "Crohn disease diagnosis treatment",
    "Crohn's disease pathogenesis",
    "Crohn disease machine learning",
    "Crohn's disease biomarkers",
    "Crohn disease inflammation",
    "Crohn's disease gut microbiome",
    "Crohn disease biologics",
    "Crohn's disease treatment response",
    "Crohn disease remission",
    "Crohn's disease genetics",
    "Crohn disease imaging",
    "Crohn's disease prediction",
    "inflammatory bowel disease Crohn",
    "Crohn disease nutrition diet",
    
    # Autoimmune disease research
    "autoimmune disease diagnosis treatment",
    "systemic lupus erythematosus rheumatoid arthritis",
    "autoimmune disease machine learning",
    "immune system dysfunction",
    "autoimmunity pathogenesis",
    
    # Treatment and drugs
    "autoimmune disease treatment drugs",
    "immunosuppressive therapy autoimmune",
    "biologics autoimmune disease",
    "DMARDs rheumatoid arthritis",
    "autoimmune disease medication",
    
    # Vagus nerve and immune system
    "vagus nerve immune system",
    "vagus nerve inflammation",
    "vagal nerve stimulation autoimmune",
    "cholinergic anti-inflammatory pathway",
    "vagus nerve immunomodulation",
    
    # Fasting and immune system
    "fasting immune system",
    "intermittent fasting autoimmune",
    "fasting inflammation",
    "caloric restriction immune function",
    "autophagy immune system",
    
    # Diet and immune system (including carnivore/lion diet)
    "carnivore diet autoimmune",
    "lion diet inflammation",
    "ketogenic diet immune system",
    "diet autoimmune disease",
    "nutritional immunology",
    "elimination diet autoimmune",
    
    # General immune research
    "immune system machine learning",
    "immunology artificial intelligence",
    "immune response prediction",
    "inflammatory markers autoimmune",
    "cytokines autoimmune disease",
]


# =============================================================================
# PDF DOWNLOAD FUNCTIONS
# =============================================================================

def get_pdf_dir() -> Path:
    """Get or create PDF directory."""
    pdf_dir = Config.DATA_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


def download_pdf(url: str, filepath: Path, timeout: int = PDF_DOWNLOAD_TIMEOUT, max_redirects: int = 5) -> bool:
    """
    Download a PDF from URL with improved error handling and redirect support.
    
    Args:
        url: PDF URL
        filepath: Path to save PDF
        timeout: Download timeout in seconds
        max_redirects: Maximum number of redirects to follow
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.ncbi.nlm.nih.gov/' if 'ncbi.nlm.nih.gov' in url else None
        }
        
        # Remove None values
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Use session to handle cookies and redirects
        session = requests.Session()
        session.max_redirects = max_redirects
        
        response = session.get(
            url, 
            headers=headers, 
            timeout=timeout, 
            stream=True,
            allow_redirects=True
        )
        
        # Check status
        if response.status_code != 200:
            return False
        
        # Check content type - be more lenient
        content_type = response.headers.get('content-type', '').lower()
        is_pdf_content = (
            'pdf' in content_type or 
            url.endswith('.pdf') or
            'application/octet-stream' in content_type
        )
        
        # Download the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify file is valid PDF
        file_size = filepath.stat().st_size
        
        # Check minimum size
        if file_size < 1000:
            filepath.unlink()
            return False
        
        # Try to verify it's actually a PDF by checking magic bytes
        try:
            with open(filepath, 'rb') as f:
                magic_bytes = f.read(4)
                # PDF files start with %PDF
                if not magic_bytes.startswith(b'%PDF'):
                    # Some PDFs might have BOM or whitespace, check first 1024 bytes
                    f.seek(0)
                    first_kb = f.read(1024)
                    if b'%PDF' not in first_kb:
                        filepath.unlink()
                        return False
        except Exception:
            # If we can't verify, assume it's OK if content-type suggests PDF
            if not is_pdf_content:
                filepath.unlink()
                return False
        
        return True
        
    except requests.exceptions.TooManyRedirects as e:
        if filepath.exists():
            filepath.unlink()
        return False
    except requests.exceptions.Timeout as e:
        if filepath.exists():
            filepath.unlink()
        return False
    except requests.exceptions.HTTPError as e:
        # 403/404 are common for paywalled papers
        if filepath.exists():
            filepath.unlink()
        return False
    except requests.exceptions.RequestException as e:
        if filepath.exists():
            filepath.unlink()
        return False
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        return False


def get_pmc_id_from_record(record: dict) -> Optional[str]:
    """
    Extract PMC ID directly from PubMed record.
    
    Args:
        record: PubMed record dictionary
    
    Returns:
        PMC ID if found, None otherwise
    """
    try:
        # Check ArticleIdList for PMC ID
        article_ids = record.get("PubmedData", {}).get("ArticleIdList", [])
        for article_id in article_ids:
            if article_id.attributes.get("IdType") == "pmc":
                return article_id.strip()
        
        # Also check MedlineCitation
        medline = record.get("MedlineCitation", {})
        article_ids = medline.get("ArticleIdList", [])
        for article_id in article_ids:
            if hasattr(article_id, 'attributes') and article_id.attributes.get("IdType") == "pmc":
                return str(article_id).strip()
            elif isinstance(article_id, dict) and article_id.get("@IdType") == "pmc":
                return str(article_id.get("#text", "")).strip()
        
        return None
    except Exception:
        return None


def get_pmc_pdf_url(pmid: str, pmc_id: Optional[str] = None, logger=None) -> Optional[str]:
    """
    Get PDF URL from PubMed Central for a given PMID.
    Tries multiple URL formats.
    
    Args:
        pmid: PubMed ID
        pmc_id: Optional PMC ID (if already known)
        logger: Optional logger
    
    Returns:
        PDF URL if available, None otherwise
    """
    try:
        # If PMC ID not provided, try to get it
        if not pmc_id:
            Entrez.email = Config.PUBMED_EMAIL
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(handle)
            handle.close()
            
            # Extract PMC ID
            for linkset in record:
                for link in linkset.get("LinkSetDb", []):
                    if link.get("DbTo") == "pmc":
                        for linkid in link.get("Link", []):
                            pmc_id = linkid.get("Id")
                            break
        
        if pmc_id:
            # Clean PMC ID (remove 'PMC' prefix if present)
            pmc_id = str(pmc_id).replace("PMC", "").strip()
            
            # Try multiple PMC PDF URL formats
            # Format 1: Standard PMC PDF URL
            url1 = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
            
            # Format 2: Direct PDF download (sometimes works better)
            url2 = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/main.pdf"
            
            # Format 3: Alternative format
            url3 = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/PMC{pmc_id}.pdf"
            
            # Return the standard format first, download function will try alternatives if needed
            return url1
        
        return None
        
    except Exception as e:
        if logger:
            logger.debug(f"Error getting PMC URL for {pmid}: {e}")
        return None


def try_multiple_pmc_urls(pmc_id: str) -> List[str]:
    """
    Generate multiple PMC URL formats to try.
    
    Args:
        pmc_id: PMC ID (without 'PMC' prefix)
    
    Returns:
        List of URLs to try in order
    """
    pmc_id = str(pmc_id).replace("PMC", "").strip()
    return [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/main.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/PMC{pmc_id}.pdf",
    ]


def get_unpaywall_pdf_url(doi: str, logger=None) -> Optional[str]:
    """
    Get open access PDF URL from Unpaywall API.
    
    Args:
        doi: DOI of the paper
        logger: Optional logger
    
    Returns:
        PDF URL if available, None otherwise
    """
    if not doi:
        return None
    
    try:
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
        
        # Unpaywall API (free, no API key needed for basic use)
        url = f"https://api.unpaywall.org/v2/{doi}?email={Config.PUBMED_EMAIL or 'anonymous@example.com'}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if paper is open access
            if data.get("is_oa", False):
                # Get best PDF URL
                best_pdf = data.get("best_oa_location", {})
                pdf_url = best_pdf.get("url_for_pdf")
                
                if pdf_url:
                    return pdf_url
        
        return None
        
    except Exception as e:
        if logger:
            logger.debug(f"Error checking Unpaywall for {doi}: {e}")
        return None


def extract_doi_from_record(record: dict) -> Optional[str]:
    """
    Extract DOI from PubMed record.
    
    Args:
        record: PubMed record dictionary
    
    Returns:
        DOI if found, None otherwise
    """
    try:
        # Check ArticleIdList for DOI
        article_ids = record.get("PubmedData", {}).get("ArticleIdList", [])
        for article_id in article_ids:
            if article_id.attributes.get("IdType") == "doi":
                return str(article_id).strip()
        
        # Also check MedlineCitation
        medline = record.get("MedlineCitation", {})
        article_ids = medline.get("ArticleIdList", [])
        for article_id in article_ids:
            if hasattr(article_id, 'attributes') and article_id.attributes.get("IdType") == "doi":
                return str(article_id).strip()
            elif isinstance(article_id, dict) and article_id.get("@IdType") == "doi":
                return str(article_id.get("#text", "")).strip()
        
        return None
    except Exception:
        return None


def get_arxiv_pdf_url(arxiv_id: str) -> str:
    """Get PDF URL for arXiv paper."""
    # Clean the arxiv_id
    arxiv_id = arxiv_id.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


# =============================================================================
# PDF TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text or None if extraction fails
    """
    if not PDF_SUPPORT:
        return None
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        # Clean up the text
        full_text = clean_extracted_text(full_text)
        
        return full_text if len(full_text) > 500 else None
        
    except Exception:
        return None


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted PDF text.
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove reference numbers like [1], [2,3], etc.
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    
    # Remove figure/table references
    text = re.sub(r'(?:Figure|Fig\.|Table)\s*\d+[a-z]?', '', text, flags=re.IGNORECASE)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text.strip()


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_start = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            # Save current chunk if it's large enough
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": current_chunk.strip(),
                    "char_start": chunk_start,
                    "char_end": chunk_start + len(current_chunk)
                })
            
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                # Get last 'overlap' characters as context
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                # Find a good break point (sentence or word boundary)
                break_point = overlap_text.rfind('. ')
                if break_point == -1:
                    break_point = overlap_text.rfind(' ')
                if break_point > 0:
                    overlap_text = overlap_text[break_point+1:].strip()
                
                current_chunk = overlap_text + "\n\n" + para
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(para) - 2
            else:
                current_chunk = para
                chunk_start = chunk_start + len(current_chunk)
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append({
            "text": current_chunk.strip(),
            "char_start": chunk_start,
            "char_end": chunk_start + len(current_chunk)
        })
    
    return chunks


def identify_section(text: str) -> str:
    """
    Identify the section type of a chunk based on content.
    
    Args:
        text: Chunk text
    
    Returns:
        Section type string
    """
    text_lower = text.lower()[:500]  # Check first 500 chars
    
    if any(word in text_lower for word in ['abstract', 'summary']):
        return "abstract"
    elif any(word in text_lower for word in ['introduction', 'background']):
        return "introduction"
    elif any(word in text_lower for word in ['method', 'materials', 'procedure', 'protocol']):
        return "methods"
    elif any(word in text_lower for word in ['result', 'finding', 'outcome']):
        return "results"
    elif any(word in text_lower for word in ['discussion', 'interpretation']):
        return "discussion"
    elif any(word in text_lower for word in ['conclusion', 'summary']):
        return "conclusion"
    elif any(word in text_lower for word in ['reference', 'bibliography', 'citation']):
        return "references"
    else:
        return "body"


# =============================================================================
# PUBMED FUNCTIONS
# =============================================================================

def search_pubmed(query: str, max_results: int = MAX_PAPERS_PER_QUERY_PUBMED, logger=None) -> List[Dict]:
    """Search PubMed for papers."""
    if not Config.PUBMED_EMAIL or Config.PUBMED_EMAIL == "your.email@example.com":
        if logger:
            logger.warning("PUBMED_EMAIL not configured, skipping PubMed search")
        return []
    
    Entrez.email = Config.PUBMED_EMAIL
    papers = []
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        
        if not pmids:
            return papers
        
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        for record in records.get("PubmedArticle", []):
            try:
                paper = _parse_pubmed_record(record)
                if paper and paper.get("abstract") and len(paper["abstract"]) >= 100:
                    papers.append(paper)
            except Exception as e:
                if logger:
                    logger.debug(f"Error parsing PubMed record: {e}")
                continue
        
        time.sleep(0.5)
        
    except Exception as e:
        if logger:
            logger.error(f"Error searching PubMed: {e}")
    
    return papers


def _parse_pubmed_record(record: dict) -> Dict:
    """Parse a PubMed record into paper dictionary."""
    medline = record.get("MedlineCitation", {})
    article = medline.get("Article", {})
    
    title = article.get("ArticleTitle", "")
    
    abstract_list = article.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract_list, list):
        abstract = " ".join([str(a) for a in abstract_list])
    else:
        abstract = str(abstract_list) if abstract_list else ""
    
    journal = article.get("Journal", {}).get("Title", "")
    
    pub_date_node = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = pub_date_node.get("Year", "")
    month = pub_date_node.get("Month", "")
    pub_date = f"{year}-{month}" if year else ""
    
    authors = []
    for author in article.get("AuthorList", [])[:5]:
        last_name = author.get("LastName", "")
        first_name = author.get("ForeName", "")
        if last_name:
            authors.append(f"{first_name} {last_name}".strip())
    
    pmid = str(medline.get("PMID", ""))
    
    # Extract PMC ID and DOI from record
    pmc_id = get_pmc_id_from_record(record)
    doi = extract_doi_from_record(record)
    
    # Try to get PMC PDF URL if PMC ID is available
    pdf_url = None
    if pmc_id:
        pdf_url = get_pmc_pdf_url(pmid, pmc_id)
    
    return {
        "source": "PubMed",
        "pmid": pmid,
        "pmc_id": pmc_id,
        "doi": doi,
        "arxiv_id": "",
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "pub_date": pub_date,
        "authors": authors,
        "pdf_url": pdf_url,
        "pdf_path": None,
        "full_text": None
    }


# =============================================================================
# ARXIV FUNCTIONS
# =============================================================================

def search_arxiv(query: str, max_results: int = MAX_PAPERS_PER_QUERY_ARXIV, logger=None) -> List[Dict]:
    """Search arXiv for papers."""
    papers = []
    
    try:
        # Use new Client API to avoid deprecation warning
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Use client.results() instead of search.results()
        for result in client.results(search):
            if not result.summary or len(result.summary) < 100:
                continue
            
            arxiv_id = result.entry_id.split('/')[-1]
            
            paper = {
                "source": "arXiv",
                "pmid": "",
                "pmc_id": None,
                "doi": None,
                "arxiv_id": arxiv_id,
                "title": result.title,
                "abstract": result.summary,
                "journal": "arXiv",
                "pub_date": result.published.strftime("%Y-%m-%d") if result.published else "",
                "authors": [author.name for author in result.authors[:5]],
                "pdf_url": get_arxiv_pdf_url(arxiv_id),
                "pdf_path": None,
                "full_text": None
            }
            
            papers.append(paper)
        
        time.sleep(0.5)
        
    except Exception as e:
        if logger:
            logger.error(f"Error searching arXiv: {e}")
    
    return papers


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def remove_duplicates(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers by title similarity."""
    seen_titles = set()
    unique_papers = []
    
    for paper in papers:
        title_lower = paper['title'].lower().strip()
        title_normalized = ''.join(c for c in title_lower if c.isalnum() or c.isspace())
        
        if title_normalized not in seen_titles:
            seen_titles.add(title_normalized)
            unique_papers.append(paper)
    
    return unique_papers


def download_paper_pdfs(papers: List[Dict], logger) -> List[Dict]:
    """
    Download PDFs for all papers that have available URLs.
    Tries multiple sources: PMC, Unpaywall (via DOI), arXiv.
    
    Args:
        papers: List of paper dictionaries
        logger: Logger instance
    
    Returns:
        Updated papers list with PDF paths
    """
    pdf_dir = get_pdf_dir()
    downloaded = 0
    failed = 0
    no_pdf = 0
    
    # Separate papers by source for better reporting
    arxiv_papers = [p for p in papers if p.get('source') == 'arXiv']
    pubmed_papers = [p for p in papers if p.get('source') == 'PubMed']
    
    print_section("DOWNLOADING PDFs")
    print(f"Total papers: {len(papers)} ({len(arxiv_papers)} arXiv, {len(pubmed_papers)} PubMed)")
    print("Priority: arXiv (100% PDF availability) → PubMed Central → Unpaywall")
    print("Note: arXiv papers should all download successfully!")
    
    for i, paper in enumerate(papers):
        paper_id = paper.get('arxiv_id') or paper.get('pmid', f'paper_{i}')
        
        # Generate safe filename
        safe_id = re.sub(r'[^\w\-.]', '_', paper_id)
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        
        # Skip if already downloaded
        if pdf_path.exists():
            paper['pdf_path'] = str(pdf_path)
            downloaded += 1
            continue
        
        pdf_url = None
        source_name = None
        
        # Try 1: Check if PDF URL already set (from PMC during parsing)
        if paper.get('pdf_url'):
            pdf_url = paper['pdf_url']
            source_name = "PMC"
        
        # Try 2: Get PMC URL if not already set
        elif paper.get('pmid') and paper.get('pmc_id'):
            pdf_url = get_pmc_pdf_url(paper['pmid'], paper['pmc_id'], logger)
            source_name = "PMC"
        
        # Try 3: Try Unpaywall API if DOI is available
        elif paper.get('doi'):
            pdf_url = get_unpaywall_pdf_url(paper['doi'], logger)
            source_name = "Unpaywall"
        
        # Try 4: For arXiv papers, URL should already be set
        elif paper.get('source') == 'arXiv' and paper.get('pdf_url'):
            pdf_url = paper['pdf_url']
            source_name = "arXiv"
        
        if pdf_url:
            print(f"  [{i+1}/{len(papers)}] Downloading ({source_name}): {paper['title'][:50]}...")
            
            # For PMC, try multiple URL formats
            urls_to_try = [pdf_url]
            if source_name == "PMC" and paper.get('pmc_id'):
                urls_to_try.extend(try_multiple_pmc_urls(paper['pmc_id']))
                urls_to_try = list(dict.fromkeys(urls_to_try))  # Remove duplicates
            
            success = False
            for url_attempt in urls_to_try:
                if download_pdf(url_attempt, pdf_path):
                    paper['pdf_path'] = str(pdf_path)
                    paper['pdf_url'] = url_attempt  # Save the working URL
                    downloaded += 1
                    success = True
                    print(f"    ✓ Downloaded from {source_name}")
                    logger.info(f"Downloaded PDF from {source_name}: {paper_id}")
                    break
            
            if not success:
                failed += 1
                print(f"    ✗ Failed from {source_name} (tried {len(urls_to_try)} URL(s))")
                logger.warning(f"Failed to download PDF from {source_name}: {paper_id}")
            
            time.sleep(1)  # Rate limiting
        else:
            no_pdf += 1
            reason = []
            if paper.get('source') == 'PubMed':
                if not paper.get('pmc_id'):
                    reason.append("not in PMC")
                if not paper.get('doi'):
                    reason.append("no DOI")
                elif not paper.get('pdf_url'):
                    reason.append("not open access")
            reason_str = f" ({', '.join(reason)})" if reason else ""
            print(f"  [{i+1}/{len(papers)}] No PDF available{reason_str}: {paper['title'][:50]}...")
    
    # Calculate success rates by source
    arxiv_downloaded = sum(1 for p in papers if p.get('source') == 'arXiv' and p.get('pdf_path'))
    pubmed_downloaded = sum(1 for p in papers if p.get('source') == 'PubMed' and p.get('pdf_path'))
    
    print(f"\n✓ Download Summary:")
    print(f"  Total Downloaded: {downloaded}")
    print(f"    arXiv: {arxiv_downloaded}/{len(arxiv_papers)} ({100*arxiv_downloaded/max(len(arxiv_papers),1):.1f}% success)")
    print(f"    PubMed: {pubmed_downloaded}/{len(pubmed_papers)} ({100*pubmed_downloaded/max(len(pubmed_papers),1):.1f}% success)")
    print(f"  Failed: {failed}")
    print(f"  No PDF available: {no_pdf}")
    logger.info(f"PDF download complete: {downloaded} downloaded ({arxiv_downloaded} arXiv, {pubmed_downloaded} PubMed), {failed} failed, {no_pdf} no PDF")
    
    return papers


def extract_paper_texts(papers: List[Dict], logger) -> List[Dict]:
    """
    Extract text from downloaded PDFs.
    
    Args:
        papers: List of paper dictionaries
        logger: Logger instance
    
    Returns:
        Updated papers list with extracted text
    """
    if not PDF_SUPPORT:
        print("⚠ PDF extraction not available. Install PyMuPDF: pip install pymupdf")
        return papers
    
    print_section("EXTRACTING TEXT FROM PDFs")
    
    extracted = 0
    
    for i, paper in enumerate(papers):
        pdf_path = paper.get('pdf_path')
        
        if pdf_path and Path(pdf_path).exists():
            print(f"  [{i+1}/{len(papers)}] Extracting: {paper['title'][:50]}...")
            
            text = extract_text_from_pdf(Path(pdf_path))
            
            if text and len(text) > 1000:
                paper['full_text'] = text
                extracted += 1
                print(f"    ✓ Extracted {len(text):,} characters")
                logger.info(f"Extracted text from: {paper.get('arxiv_id') or paper.get('pmid')}")
            else:
                print(f"    ✗ Extraction failed or text too short")
        else:
            # Use abstract as fallback
            if paper.get('abstract'):
                paper['full_text'] = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    
    print(f"\n✓ Extracted text from {extracted} PDFs")
    logger.info(f"Text extraction complete: {extracted} papers")
    
    return papers


def chunk_all_papers(papers: List[Dict], logger) -> List[Dict]:
    """
    Chunk all papers into training segments.
    
    Args:
        papers: List of paper dictionaries
        logger: Logger instance
    
    Returns:
        List of chunk dictionaries
    """
    print_section("CHUNKING PAPERS")
    
    all_chunks = []
    
    for paper in papers:
        text = paper.get('full_text')
        
        if not text:
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        
        for j, chunk in enumerate(chunks):
            # Identify section type
            section = identify_section(chunk['text'])
            
            # Skip reference sections
            if section == "references":
                continue
            
            chunk_data = {
                "paper_id": paper.get('arxiv_id') or paper.get('pmid', ''),
                "paper_title": paper['title'],
                "source": paper['source'],
                "chunk_index": j,
                "total_chunks": len(chunks),
                "section": section,
                "text": chunk['text'],
                "char_start": chunk['char_start'],
                "char_end": chunk['char_end']
            }
            
            all_chunks.append(chunk_data)
    
    print(f"✓ Created {len(all_chunks)} chunks from {len(papers)} papers")
    logger.info(f"Chunking complete: {len(all_chunks)} chunks")
    
    return all_chunks


def create_training_data(chunks: List[Dict], papers: List[Dict], logger) -> List[Dict]:
    """
    Create training data from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        papers: List of paper dictionaries (for abstracts)
        logger: Logger instance
    
    Returns:
        List of training examples
    """
    print_section("CREATING TRAINING DATA")
    
    training_data = []
    
    # Create a lookup for paper abstracts
    paper_abstracts = {
        (p.get('arxiv_id') or p.get('pmid', '')): p.get('abstract', '')
        for p in papers
    }
    
    for chunk in chunks:
        paper_id = chunk['paper_id']
        paper_title = chunk['paper_title']
        section = chunk['section']
        text = chunk['text']
        
        # Create different training formats based on section type
        if section == "abstract":
            # For abstracts, create a summarization task
            instruction = f"""Analyze this research abstract about autoimmune disease and identify the key findings:

Title: {paper_title}

{text}"""
        
        elif section in ["introduction", "background"]:
            # For introduction, create a context understanding task
            instruction = f"""Based on this introduction from a research paper on autoimmune disease, explain the background and motivation:

Title: {paper_title}

{text}"""
        
        elif section == "methods":
            # For methods, create a methodology explanation task
            instruction = f"""Explain the research methodology described in this section about autoimmune disease research:

Title: {paper_title}

{text}"""
        
        elif section == "results":
            # For results, create an interpretation task
            instruction = f"""Interpret these research findings about autoimmune disease:

Title: {paper_title}

{text}"""
        
        elif section in ["discussion", "conclusion"]:
            # For discussion/conclusion, create an implications task
            instruction = f"""Discuss the implications of these findings for autoimmune disease diagnosis and treatment:

Title: {paper_title}

{text}"""
        
        else:
            # General body text
            instruction = f"""Analyze this excerpt from a research paper on autoimmune disease and explain its relevance to clinical practice:

Title: {paper_title}

{text}"""
        
        # Format as training example
        training_text = format_prompt(instruction)
        
        training_data.append({
            "text": training_text,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "source": chunk['source'],
            "section": section,
            "chunk_index": chunk['chunk_index']
        })
    
    print(f"✓ Created {len(training_data)} training examples")
    logger.info(f"Training data created: {len(training_data)} examples")
    
    return training_data


def download_all_papers(logger) -> List[Dict]:
    """Download papers from all sources. Prioritizes arXiv (primary source)."""
    all_papers = []
    
    print_section("SEARCHING FOR RESEARCH PAPERS")
    print("🎯 PRIMARY FOCUS: Rheumatoid Arthritis (RA) & Crohn's Disease")
    print("📚 Additional topics: Immune system, autoimmune disease, treatment, vagus nerve, fasting, diet")
    print("📥 Primary source: arXiv (all papers have free PDFs)")
    
    # arXiv papers FIRST (PRIMARY SOURCE - all have free PDFs)
    print("\n" + "="*60)
    print("SEARCHING ARXIV (PRIMARY SOURCE)")
    print("="*60)
    for i, query in enumerate(ARXIV_QUERIES, 1):
        print(f"  [{i}/{len(ARXIV_QUERIES)}] {query}")
        papers = search_arxiv(query, max_results=MAX_PAPERS_PER_QUERY_ARXIV, logger=logger)
        all_papers.extend(papers)
        print(f"    Found {len(papers)} papers")
        logger.info(f"arXiv query {i}: {len(papers)} papers")
    
    # PubMed papers SECOND (supplementary, many don't have PDFs)
    print("\n" + "="*60)
    print("SEARCHING PUBMED (SUPPLEMENTARY - Open Access Only)")
    print("="*60)
    for i, query in enumerate(PUBMED_QUERIES, 1):
        print(f"  [{i}/{len(PUBMED_QUERIES)}] {query[:70]}...")
        papers = search_pubmed(query, max_results=MAX_PAPERS_PER_QUERY_PUBMED, logger=logger)
        all_papers.extend(papers)
        print(f"    Found {len(papers)} papers")
        logger.info(f"PubMed query {i}: {len(papers)} papers")
    
    # Remove duplicates (keep arXiv papers if duplicate found)
    print(f"\nRemoving duplicates...")
    print(f"  Before: {len(all_papers)} papers")
    
    # Prioritize arXiv papers in deduplication
    arxiv_papers = [p for p in all_papers if p.get('source') == 'arXiv']
    pubmed_papers = [p for p in all_papers if p.get('source') == 'PubMed']
    
    # Remove duplicates from PubMed that exist in arXiv
    seen_titles = set()
    unique_papers = []
    
    # Add arXiv papers first
    for paper in arxiv_papers:
        title_normalized = ''.join(c for c in paper['title'].lower().strip() if c.isalnum() or c.isspace())
        if title_normalized not in seen_titles:
            seen_titles.add(title_normalized)
            unique_papers.append(paper)
    
    # Add PubMed papers that aren't duplicates
    for paper in pubmed_papers:
        title_normalized = ''.join(c for c in paper['title'].lower().strip() if c.isalnum() or c.isspace())
        if title_normalized not in seen_titles:
            seen_titles.add(title_normalized)
            unique_papers.append(paper)
    
    all_papers = unique_papers
    print(f"  After: {len(all_papers)} papers")
    print(f"    arXiv: {len([p for p in all_papers if p.get('source') == 'arXiv'])}")
    print(f"    PubMed: {len([p for p in all_papers if p.get('source') == 'PubMed'])}")
    logger.info(f"After deduplication: {len(all_papers)} papers ({len(arxiv_papers)} arXiv, {len(pubmed_papers)} PubMed)")
    
    return all_papers


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print_header("SCRIPT 3: DOWNLOAD & PROCESS RESEARCH PAPERS (Full PDF)")
    
    # Setup
    logger = setup_logging("3_download_papers")
    Config.ensure_directories()
    
    # Check PDF support
    if not PDF_SUPPORT:
        print("\n⚠ WARNING: PyMuPDF not installed.")
        print("  Install with: pip install pymupdf")
        print("  Continuing with abstract-only mode...\n")
        logger.warning("PyMuPDF not installed, using abstract-only mode")
    
    # Validate config
    warnings = Config.validate()
    for warning in warnings:
        print(f"⚠ WARNING: {warning}")
        logger.warning(warning)
    
    try:
        # Step 1: Search and download paper metadata
        papers = download_all_papers(logger)
        
        if not papers:
            print("\n⚠ WARNING: No papers found.")
            print("Check your PUBMED_EMAIL setting in .env file.")
            logger.warning("No papers downloaded")
            sys.exit(1)
        
        # Step 2: Download PDFs
        papers = download_paper_pdfs(papers, logger)
        
        # Step 3: Extract text from PDFs
        papers = extract_paper_texts(papers, logger)
        
        # Save raw papers (without full text to save space)
        raw_papers = [{k: v for k, v in p.items() if k != 'full_text'} for p in papers]
        raw_file = Config.DATA_DIR / "raw_papers.json"
        save_json(raw_papers, raw_file)
        print(f"\n✓ Raw papers saved to {raw_file}")
        logger.info(f"Raw papers saved: {len(papers)}")
        
        # Step 4: Chunk papers
        chunks = chunk_all_papers(papers, logger)
        
        # Save chunks
        chunks_file = Config.DATA_DIR / "paper_chunks.json"
        save_json(chunks, chunks_file)
        print(f"✓ Chunks saved to {chunks_file}")
        
        # Step 5: Create training data
        training_data = create_training_data(chunks, papers, logger)
        
        # Save training data
        training_file = Config.DATA_DIR / "papers_training_data.json"
        save_json(training_data, training_file)
        print(f"✓ Training data saved to {training_file}")
        logger.info(f"Training data saved: {len(training_data)}")
        
        # Print summary
        print_section("PROCESSING SUMMARY")
        print(f"Papers Found: {len(papers)}")
        
        papers_with_pdf = sum(1 for p in papers if p.get('pdf_path'))
        papers_with_text = sum(1 for p in papers if p.get('full_text'))
        
        # Count RA and Crohn's papers (PRIMARY FOCUS)
        ra_papers = []
        crohns_papers = []
        for paper in papers:
            title_lower = paper.get('title', '').lower()
            abstract_lower = paper.get('abstract', '').lower()
            text_lower = paper.get('full_text', '').lower() if paper.get('full_text') else ''
            combined = title_lower + ' ' + abstract_lower + ' ' + text_lower
            
            # Check for RA
            if any(term in combined for term in ['rheumatoid arthritis', 'rheumatoid arthrit', 'ra patients', 'ra disease']):
                ra_papers.append(paper)
            
            # Check for Crohn's
            if any(term in combined for term in ["crohn's disease", "crohn disease", "crohns disease", 'crohn\'s']):
                crohns_papers.append(paper)
        
        ra_with_pdf = sum(1 for p in ra_papers if p.get('pdf_path'))
        crohns_with_pdf = sum(1 for p in crohns_papers if p.get('pdf_path'))
        
        print(f"\n📊 PRIMARY FOCUS AREAS:")
        print(f"  Rheumatoid Arthritis (RA): {len(ra_papers)} papers ({ra_with_pdf} with PDFs)")
        print(f"  Crohn's Disease: {len(crohns_papers)} papers ({crohns_with_pdf} with PDFs)")
        
        print(f"\n📄 OVERALL STATISTICS:")
        print(f"  PDFs Downloaded: {papers_with_pdf}")
        print(f"  Text Extracted: {papers_with_text}")
        print(f"  Total Chunks: {len(chunks)}")
        print(f"  Training Examples: {len(training_data)}")
        
        # Section breakdown
        section_counts = {}
        for chunk in chunks:
            section = chunk['section']
            section_counts[section] = section_counts.get(section, 0) + 1
        
        print(f"\n📑 Chunks by Section:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count}")
        
        print_header("✓ Script 3 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
