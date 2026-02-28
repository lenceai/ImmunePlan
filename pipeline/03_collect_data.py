#!/usr/bin/env python3
"""
Step 03: Collect Training Data
Book: Chapters 3-4 prep — Download research papers, extract text, chunk for RAG & fine-tuning.

Standalone: conda run -n base python pipeline/03_collect_data.py
Output:     data/raw_papers.json, data/paper_chunks.json, data/papers_training_data.json

PDF Strategy:
  - arXiv papers: download full PDF → fitz extraction → section-aware chunking
  - PubMed papers: abstract only (no open PDF)
  - Each paper's abstract is stored as a special reference chunk (is_abstract_ref=True)
  - Body chunks embed the abstract as [Reference Abstract] context in the training prompt
"""
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, save_json, load_json, print_step, DATA_DIR, PUBMED_EMAIL,
    ensure_directories, format_prompt,
)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 500
PDF_CACHE_DIR = DATA_DIR / "pdfs"
PDF_DOWNLOAD_DELAY = 3.0   # seconds between downloads (respect arXiv rate limit)
MAX_PDF_MB = 20

DISCLAIMER = (
    "\n\n**Important**: This information is for educational purposes only "
    "and is NOT a substitute for professional medical advice. "
    "Always consult your healthcare provider."
)

# ---------------------------------------------------------------------------
# Medical Q&A templates — condition-centric questions matching evaluation format
# ---------------------------------------------------------------------------

# Partial string matches to detect conditions in chunk text
CONDITION_KEYWORDS = {
    "rheumatoid_arthritis": [
        "rheumatoid arthritis", " RA ", " RA,", " RA.", "anti-CCP", "DAS28",
        "joint inflammation", "synovitis", "pannus",
    ],
    "crohns": [
        "crohn", "inflammatory bowel", " IBD", "crohn's", "ileitis",
        "intestinal inflammation", "calprotectin",
    ],
    "lupus": [
        "lupus", " SLE ", "systemic lupus", "anti-dsDNA", "anti-Smith",
        "lupus nephritis", "malar rash",
    ],
    "sjogrens": [
        "sjogren", "sicca", "xerostomia", "anti-SSA", "anti-SSB",
        "anti-Ro", "anti-La", "dry eye", "dry mouth",
    ],
}

# Keywords that signal what topic the chunk is about
TOPIC_KEYWORDS = {
    "diagnosis": [
        "diagnos", "criteria", "classification", "ACR criteria", "EULAR",
        "imaging", "serology", "assessment", "scoring",
    ],
    "biomarkers": [
        "biomarker", "antibod", "anti-CCP", "rheumatoid factor",
        "ANA", "anti-dsDNA", "calprotectin", "CRP", "ESR", "laboratory",
        "serolog", "immunoglobulin",
    ],
    "treatment": [
        "treatment", "therapy", "DMARD", "biologic", "methotrexate",
        "adalimumab", "infliximab", "etanercept", "rituximab", "JAK inhibitor",
        "remission", "manage", "vedolizumab", "ustekinumab", "hydroxychloroquine",
    ],
    "pathogenesis": [
        "pathogen", "mechanism", "immune", "cytokine", "interleukin",
        "T cell", "B cell", "autoantibod", "complement", "inflammation",
    ],
}

# Medical Q&A templates per condition × topic — matching evaluation question style
MEDICAL_QA_TEMPLATES = {
    "rheumatoid_arthritis": {
        "diagnosis": [
            "What are the diagnostic criteria for rheumatoid arthritis?",
            "How is rheumatoid arthritis clinically diagnosed?",
            "What clinical and laboratory findings confirm a diagnosis of RA?",
        ],
        "biomarkers": [
            "What biomarkers are used in rheumatoid arthritis diagnosis?",
            "What is the role of anti-CCP antibodies and rheumatoid factor in RA?",
            "Which serological markers help distinguish RA from other arthritides?",
        ],
        "treatment": [
            "What are the treatment options for rheumatoid arthritis?",
            "How are DMARDs used in rheumatoid arthritis management?",
            "What biologic therapies are used for rheumatoid arthritis and when?",
        ],
        "pathogenesis": [
            "What is the pathogenesis of rheumatoid arthritis?",
            "How does the immune system drive joint damage in RA?",
        ],
        "general": [
            "What is rheumatoid arthritis and how does it affect joints?",
            "What are the clinical features of rheumatoid arthritis?",
        ],
    },
    "crohns": {
        "diagnosis": [
            "What are the diagnostic criteria for Crohn's disease?",
            "How is Crohn's disease diagnosed endoscopically and histologically?",
            "What investigations are used to diagnose Crohn's disease?",
        ],
        "biomarkers": [
            "What biomarkers are used in Crohn's disease monitoring?",
            "What is the role of fecal calprotectin in inflammatory bowel disease?",
            "Which serological and stool markers help monitor Crohn's disease activity?",
        ],
        "treatment": [
            "What are the treatment options for Crohn's disease?",
            "What biologic therapies are approved for Crohn's disease?",
            "How is Crohn's disease managed with immunosuppressants and biologics?",
        ],
        "pathogenesis": [
            "What is the pathogenesis of Crohn's disease?",
            "How does gut dysbiosis contribute to Crohn's disease?",
        ],
        "general": [
            "What is Crohn's disease and which parts of the GI tract does it affect?",
            "What are the symptoms and complications of Crohn's disease?",
        ],
    },
    "lupus": {
        "diagnosis": [
            "What are the diagnostic criteria for systemic lupus erythematosus (SLE)?",
            "How is lupus diagnosed using the ACR/EULAR classification criteria?",
        ],
        "biomarkers": [
            "What autoantibodies are used to diagnose lupus (SLE)?",
            "What is the role of anti-dsDNA and anti-Smith antibodies in lupus diagnosis?",
            "Which laboratory markers are most specific for systemic lupus erythematosus?",
        ],
        "treatment": [
            "What are the treatment options for systemic lupus erythematosus?",
            "How is lupus managed with hydroxychloroquine and immunosuppressants?",
        ],
        "pathogenesis": [
            "What is the pathogenesis of systemic lupus erythematosus?",
            "How does type I interferon dysregulation contribute to lupus?",
        ],
        "general": [
            "What is systemic lupus erythematosus and which organs does it affect?",
            "What are the clinical manifestations of lupus (SLE)?",
        ],
    },
    "sjogrens": {
        "diagnosis": [
            "What are the diagnostic criteria for Sjogren's syndrome?",
            "How is Sjogren's syndrome diagnosed clinically and serologically?",
        ],
        "biomarkers": [
            "What autoantibodies are associated with Sjogren's syndrome?",
            "What is the role of anti-Ro/SSA and anti-La/SSB antibodies in Sjogren's diagnosis?",
        ],
        "treatment": [
            "What are the treatment options for Sjogren's syndrome?",
            "How are sicca symptoms managed in Sjogren's syndrome?",
        ],
        "general": [
            "What is Sjogren's syndrome and what are its main symptoms?",
            "How does Sjogren's syndrome affect the exocrine glands?",
        ],
    },
}


def _detect_conditions_topics(chunk: Dict):
    """
    Detect which autoimmune conditions and clinical topics are present in a chunk.
    Returns list of (condition_key, topic_key) pairs — may be empty.
    """
    text_lower = chunk['text'].lower()
    found = []
    for condition, keywords in CONDITION_KEYWORDS.items():
        if not any(kw.lower() in text_lower for kw in keywords):
            continue
        topics_found = [
            topic for topic, kws in TOPIC_KEYWORDS.items()
            if any(kw.lower() in text_lower for kw in kws)
        ]
        if not topics_found:
            topics_found = ["general"]
        for topic in topics_found:
            found.append((condition, topic))
    return found


def generate_medical_qa_pairs(chunk: Dict) -> List[Tuple[str, str]]:
    """
    Generate medical Q&A pairs where the instruction is a clinical question
    matching the evaluation format, answered from chunk content.
    Returns list of (instruction, response) tuples.
    """
    title = chunk['paper_title']
    pairs = []

    for condition, topic in _detect_conditions_topics(chunk):
        templates = MEDICAL_QA_TEMPLATES.get(condition, {})
        questions = templates.get(topic, templates.get("general", []))
        # Take up to 2 questions per condition×topic to avoid duplication
        for q in questions[:2]:
            response = (
                f"Based on evidence from recent research ('{title}'):\n\n"
                f"{chunk['text']}{DISCLAIMER}"
            )
            pairs.append((q, response))

    return pairs


# Multiple question templates per section type → 2-3x more training examples
SECTION_QUESTIONS = {
    "abstract": [
        "Summarize the main findings of the paper: '{title}'",
        "What is the paper '{title}' about?",
        "What does the study '{title}' investigate and what are its key conclusions?",
    ],
    "introduction": [
        "What is the background and research context for the study '{title}'?",
        "Why was the study '{title}' conducted and what gap does it address?",
    ],
    "methods": [
        "What research methods were used in '{title}'?",
        "How was the study '{title}' designed and conducted?",
    ],
    "results": [
        "What were the key findings reported in '{title}'?",
        "What did the study '{title}' discover or demonstrate?",
    ],
    "discussion": [
        "What conclusions did the authors of '{title}' draw from their research?",
        "What are the clinical implications discussed in '{title}'?",
    ],
    "body": [
        "What does the medical literature say about the topic covered in '{title}'?",
        "Explain the key content from the paper '{title}'.",
    ],
}

RESPONSE_PREFIXES = {
    "abstract":     "This paper, titled '{title}', reports the following",
    "introduction": "The study '{title}' provides this background",
    "methods":      "The study '{title}' employed the following methodology",
    "results":      "The study '{title}' reports these findings",
    "discussion":   "The authors of '{title}' draw these conclusions",
    "body":         "According to '{title}'",
}

# ---------------------------------------------------------------------------
# Relevance filtering — score papers before spending time on PDFs
# ---------------------------------------------------------------------------

AUTOIMMUNE_KEYWORDS = {
    "high": [
        "rheumatoid arthritis", "crohn", "inflammatory bowel", "lupus", "sjogren",
        "psoriasis", "ankylosing spondylitis", "multiple sclerosis", "autoimmune",
        "immune-mediated", "biologics", "dmard", "anti-tnf", "b-cell", "t-cell",
        "interleukin", "cytokine", "autoantibody", "vasculitis", "uveitis",
    ],
    "medium": [
        "immune", "inflammatory", "monoclonal antibody", "immunotherapy",
        "gut microbiome", "mucosal immunity", "pathogenesis", "remission", "disease activity",
        "methotrexate", "adalimumab", "infliximab", "etanercept", "rituximab",
        "JAK inhibitor", "toll-like receptor", "complement system",
    ],
}
RELEVANCE_THRESHOLD = 0.10   # minimum keyword density to keep a paper


def relevance_score(paper: Dict) -> float:
    """Return a 0-1 score based on autoimmune keyword density in title + abstract."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    if not text.strip():
        return 0.0
    word_count = max(len(text.split()), 1)
    hits = 0
    for kw in AUTOIMMUNE_KEYWORDS["high"]:
        hits += text.count(kw) * 2
    for kw in AUTOIMMUNE_KEYWORDS["medium"]:
        hits += text.count(kw)
    # Normalise: each word contributes at most 0.05 to the score budget
    return min(hits / (word_count * 0.05), 1.0)


ARXIV_QUERIES = [
    # Rheumatoid arthritis
    "rheumatoid arthritis diagnosis treatment", "rheumatoid arthritis pathogenesis",
    "rheumatoid arthritis biomarkers", "rheumatoid arthritis DMARDs biologics",
    "rheumatoid arthritis remission", "rheumatoid arthritis anti-CCP RF diagnosis",
    "treat-to-target rheumatoid arthritis DAS28", "JAK inhibitor rheumatoid arthritis",
    "TNF inhibitor biologic DMARD autoimmune",
    # Crohn's disease — expanded to improve coverage
    "Crohn's disease inflammatory bowel", "Crohn disease diagnosis treatment",
    "Crohn's disease biomarkers calprotectin", "Crohn disease biologics infliximab",
    "Crohn's disease remission vedolizumab", "Crohn disease ustekinumab biologic",
    "inflammatory bowel disease mucosal healing", "Crohn's disease fistula perianal",
    "Crohn disease surgery stricture", "IBD biologic therapy anti-TNF",
    # Lupus, Sjogren, other autoimmune
    "systemic lupus erythematosus diagnosis treatment",
    "Sjogren syndrome diagnosis xerostomia xerophthalmia",
    "autoimmune disease diagnosis treatment", "autoimmune disease machine learning",
    "immune system dysfunction", "vagus nerve immune system",
    "fasting immune system", "diet autoimmune disease",
    "psoriatic arthritis treatment biologic",
    "ankylosing spondylitis axial spondyloarthritis biologic",
]

PUBMED_QUERIES = [
    "rheumatoid arthritis[Title] AND free full text[sb] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "Crohn disease[Title] AND free full text[sb] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    "autoimmune disease[Title] AND free full text[sb] AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
]

# Matches bare section headers in PDF text (with optional leading number)
_SECTION_RE = re.compile(
    r'^\s*(?:\d+\.?\d*\.?\s+)?'
    r'(abstract|introduction|background|related work|'
    r'methods?|materials?\s+and\s+methods?|experimental(?:\s+methods?)?|'
    r'results?|findings?|'
    r'discussion|conclusions?|'
    r'acknowledgements?|references?|bibliography)'
    r'\s*$',
    re.IGNORECASE,
)

_SECTION_NORM = {
    'background': 'introduction', 'related work': 'introduction',
    'materials and methods': 'methods', 'experimental': 'methods',
    'experimental methods': 'methods',
    'findings': 'results',
    'conclusions': 'discussion', 'conclusion': 'discussion',
    'bibliography': 'references', 'acknowledgements': 'acknowledgements',
    'acknowledgements': 'acknowledgements',
}


# ---------------------------------------------------------------------------
# PDF download & extraction
# ---------------------------------------------------------------------------

def download_pdf(pdf_url: str, paper_id: str, logger=None) -> Optional[Path]:
    """Download PDF to cache dir. Returns path or None if unavailable/too large."""
    try:
        import requests
    except ImportError:
        if logger:
            logger.warning("requests not installed, cannot download PDFs")
        return None

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r'[^a-zA-Z0-9_.\-]', '_', paper_id)
    pdf_path = PDF_CACHE_DIR / f"{safe_id}.pdf"

    if pdf_path.exists():
        return pdf_path

    try:
        r = requests.get(
            pdf_url, timeout=60,
            headers={"User-Agent": "ImmunePlan/1.0 (autoimmune research project)"},
        )
        if r.status_code != 200:
            if logger:
                logger.warning(f"PDF HTTP {r.status_code}: {pdf_url}")
            return None
        size_mb = len(r.content) / 1024 / 1024
        if size_mb > MAX_PDF_MB:
            if logger:
                logger.warning(f"PDF too large ({size_mb:.1f} MB), skipping: {pdf_url}")
            return None
        if len(r.content) < 1024:
            return None
        pdf_path.write_bytes(r.content)
        return pdf_path
    except Exception as e:
        if logger:
            logger.warning(f"PDF download failed ({pdf_url}): {e}")
        return None


def extract_pdf_text(pdf_path: Path, logger=None) -> Optional[str]:
    """Extract plain text from PDF using pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(p for p in pages if p.strip())
    except Exception as e:
        if logger:
            logger.warning(f"PDF extraction failed ({pdf_path.name}): {e}")
        return None


def split_pdf_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split PDF text into (section_name, content) pairs by detecting section headers.
    Falls back to a single 'body' section if no headers found.
    """
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    sections: List[Tuple[str, str]] = []
    current_name = "preamble"
    current_lines: List[str] = []

    for line in lines:
        m = _SECTION_RE.match(line.strip())
        if m:
            content = '\n'.join(current_lines).strip()
            if content and len(content) >= MIN_CHUNK_SIZE:
                sections.append((current_name, content))
            raw = m.group(1).lower().strip()
            current_name = _SECTION_NORM.get(raw, raw)
            current_lines = []
        else:
            current_lines.append(line)

    # flush last section
    content = '\n'.join(current_lines).strip()
    if content and len(content) >= MIN_CHUNK_SIZE:
        sections.append((current_name, content))

    # if no headers detected, treat whole text as body
    if not sections:
        sections = [("body", text.strip())]

    return sections


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _is_quality_chunk(text: str) -> bool:
    """Reject garbled OCR, table fragments, and header-only chunks."""
    if not text or len(text) < MIN_CHUNK_SIZE:
        return False
    words = text.split()
    if not words:
        return False
    # Reject if >40% of characters are non-alphabetic (tables, equations, garbled)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.60:
        return False
    # Reject if average word length < 2 (garbled symbols)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word_len < 2.5:
        return False
    return True


def chunk_text(text: str) -> List[Dict]:
    """Split text into overlapping chunks respecting paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    char_start = 0   # track correctly: set when we START a new chunk

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > CHUNK_SIZE:
            if _is_quality_chunk(current):
                chunks.append({"text": current.strip(), "char_start": char_start})
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else ""
            char_start += len(current)   # advance by the chunk we just closed
            current = overlap + "\n\n" + para if overlap else para
        else:
            if not current:
                char_start += 0  # first para of this chunk — start is already set
            current = current + "\n\n" + para if current else para

    if _is_quality_chunk(current):
        chunks.append({"text": current.strip(), "char_start": char_start})
    return chunks


# ---------------------------------------------------------------------------
# PubMed / arXiv search
# ---------------------------------------------------------------------------

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
            search = arxiv.Search(
                query=q, max_results=max_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for r in client.results(search):
                if r.summary and len(r.summary) >= 100:
                    aid = r.entry_id.split('/')[-1]
                    papers.append({
                        "source": "arXiv", "arxiv_id": aid, "title": r.title,
                        "abstract": r.summary,
                        "pub_date": r.published.strftime("%Y-%m-%d") if r.published else "",
                        "authors": [a.name for a in r.authors[:5]],
                        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                    })
            time.sleep(0.5)
        except Exception as e:
            if logger:
                logger.error(f"arXiv query error: {e}")
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
                    abstract = (
                        " ".join(str(a) for a in abstract_list)
                        if isinstance(abstract_list, list)
                        else str(abstract_list)
                    )
                    if abstract and len(abstract) >= 100:
                        # Extract PMC ID for full-text access
                        pmc_id = None
                        pubmed_data = rec.get("PubmedData", {})
                        for id_item in pubmed_data.get("ArticleIdList", []):
                            if getattr(id_item, "attributes", {}).get("IdType") == "pmc":
                                pmc_id = str(id_item)
                                break
                        papers.append({
                            "source": "PubMed",
                            "title": article.get("ArticleTitle", ""),
                            "abstract": abstract,
                            "pmc_id": pmc_id,
                        })
            time.sleep(0.5)
        except Exception as e:
            if logger:
                logger.error(f"PubMed query error: {e}")
    return papers


def fetch_pmc_fulltext(pmc_id: str, email: str = "", logger=None) -> Optional[str]:
    """Fetch full article XML from PubMed Central and extract paragraph text."""
    try:
        from Bio import Entrez
        import xml.etree.ElementTree as ET
        if email:
            Entrez.email = email
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
        xml_bytes = handle.read()
        handle.close()
        root = ET.fromstring(xml_bytes)
        paragraphs = []
        for elem in root.iter("p"):
            parts = []
            if elem.text:
                parts.append(elem.text.strip())
            for child in elem:
                if child.text:
                    parts.append(child.text.strip())
                if child.tail:
                    parts.append(child.tail.strip())
            txt = " ".join(p for p in parts if p)
            if len(txt) > 50:
                paragraphs.append(txt)
        full_text = "\n\n".join(paragraphs)
        return full_text if len(full_text) >= MIN_CHUNK_SIZE else None
    except Exception as e:
        if logger:
            logger.warning(f"PMC full text failed (PMC{pmc_id}): {e}")
        return None


def deduplicate(papers):
    seen, unique = set(), []
    for p in papers:
        key = ''.join(c for c in p['title'].lower() if c.isalnum())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def build_chunks_for_paper(paper: Dict, logger=None) -> List[Dict]:
    """
    Returns all chunks for one paper.
    arXiv papers: try PDF first, fall back to abstract.
    PubMed papers: abstract only.

    Each chunk dict has:
      paper_title, source, section, text, chunk_index,
      is_abstract_ref (bool), abstract_ref (str — the paper's abstract)
    """
    title = paper['title']
    source = paper['source']
    abstract = paper.get('abstract', '')
    chunks: List[Dict] = []

    # --- 1. Abstract reference chunk (always present if abstract exists) ---
    if abstract and len(abstract) >= 100:
        for j, c in enumerate(chunk_text(abstract)):
            chunks.append({
                "paper_title": title, "source": source,
                "section": "abstract", "text": c['text'],
                "chunk_index": j, "is_abstract_ref": True,
                "abstract_ref": abstract,
            })

    # --- 2. Full-text chunks (arXiv PDF or PubMed Central) ---
    pdf_url = paper.get('pdf_url')
    arxiv_id = paper.get('arxiv_id', '')
    pmc_id = paper.get('pmc_id')
    full_text = None

    if pdf_url and arxiv_id:
        pdf_path = download_pdf(pdf_url, arxiv_id, logger=logger)
        if pdf_path:
            full_text = extract_pdf_text(pdf_path, logger=logger)
            time.sleep(PDF_DOWNLOAD_DELAY)
    elif pmc_id:
        full_text = fetch_pmc_fulltext(pmc_id, email=PUBMED_EMAIL, logger=logger)
        time.sleep(0.4)  # PMC rate limit

    if full_text:
        sections = split_pdf_sections(full_text)
        chunk_idx = 0
        for section_name, section_text in sections:
            if section_name == "references":
                continue
            if section_name == "acknowledgements":
                continue
            for c in chunk_text(section_text):
                chunks.append({
                    "paper_title": title, "source": source,
                    "section": section_name, "text": c['text'],
                    "chunk_index": chunk_idx, "is_abstract_ref": False,
                    "abstract_ref": abstract,
                })
                chunk_idx += 1
    else:
        # No PDF — if abstract already added above, nothing more to do.
        # For PubMed or failed downloads, abstract chunk is the only chunk.
        pass

    return chunks


# ---------------------------------------------------------------------------
# Training prompt builders — produce proper instruction-response pairs
# ---------------------------------------------------------------------------

def _nemotron_training_text(instruction: str, response: str) -> str:
    """Format a full instruction+response in Nemotron chat format for SFT.
    Uses /no_think — training responses are direct answers without <think> blocks.
    Must match format used in pipeline/05_finetune.py load_training_data().
    """
    return (
        f"<|im_start|>user\n/no_think {instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def generate_qa_pairs(chunk: Dict) -> List[Tuple[str, str]]:
    """
    Generate instruction-response pairs from a chunk.
    Combines:
      1. Paper-centric Q&A (section-based, keeps RAG grounding ability)
      2. Medical Q&A (condition-centric, matches evaluation question style)
    Returns list of (instruction, response) tuples.
    """
    title = chunk['paper_title']
    text = chunk['text']
    section = chunk['section']
    abstract = chunk.get('abstract_ref', '')

    # Abstract context prefix for body chunks
    abs_context = (
        f"[Reference abstract]: {abstract[:400]}...\n\n"
        if abstract and not chunk['is_abstract_ref'] else ""
    )

    # 1. Paper-centric pairs (original behaviour)
    questions = SECTION_QUESTIONS.get(section, SECTION_QUESTIONS["body"])
    prefix_template = RESPONSE_PREFIXES.get(section, "According to '{title}'")
    prefix = prefix_template.format(title=title)

    pairs = []
    for q_template in questions:
        instruction = q_template.format(title=title)
        response = f"{abs_context}{prefix}:\n\n{text}{DISCLAIMER}"
        pairs.append((instruction, response))

    # 2. Medical Q&A pairs (new — matches evaluation question format)
    pairs.extend(generate_medical_qa_pairs(chunk))

    return pairs


def build_training_examples(chunk: Dict) -> List[Dict]:
    """
    Convert a chunk into one or more training examples.
    Each example has 'text' (full SFT-formatted conversation),
    plus metadata fields.
    """
    examples = []
    for instruction, response in generate_qa_pairs(chunk):
        examples.append({
            "text": _nemotron_training_text(instruction, response),
            "instruction": instruction,
            "response": response,
            "paper_title": chunk['paper_title'],
            "source": chunk['source'],
            "section": chunk['section'],
            "is_abstract_ref": chunk['is_abstract_ref'],
        })
    return examples


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def run():
    print_step(3, "COLLECT TRAINING DATA")
    logger = setup_logging("03_collect_data")
    ensure_directories()
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fetch paper metadata ---
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

    # --- Relevance filter ---
    before = len(papers)
    papers = [p for p in papers if relevance_score(p) >= RELEVANCE_THRESHOLD]
    removed = before - len(papers)
    if removed:
        print(f"  Relevance filter: removed {removed} off-topic papers ({len(papers)} remain)")

    # --- Download PDFs and build chunks ---
    all_chunks: List[Dict] = []
    pdf_ok, pdf_fail = 0, 0

    for i, p in enumerate(papers):
        has_pdf = bool(p.get('pdf_url'))
        print(f"  [{i+1}/{len(papers)}] {p['title'][:60]}{'...' if len(p['title'])>60 else ''}"
              f"  {'[PDF]' if has_pdf else '[abstract only]'}", flush=True)
        paper_chunks = build_chunks_for_paper(p, logger=logger)
        # track PDF success
        body_chunks = [c for c in paper_chunks if not c['is_abstract_ref']]
        if has_pdf:
            if body_chunks:
                pdf_ok += 1
            else:
                pdf_fail += 1
        all_chunks.extend(paper_chunks)

    print(f"\nPDF downloads: {pdf_ok} ok, {pdf_fail} failed/skipped")

    # --- Save raw papers (no full_text field — already in PDFs cache) ---
    save_json(
        [{k: v for k, v in p.items() if k != 'full_text'} for p in papers],
        DATA_DIR / "raw_papers.json",
    )

    # --- Save chunks (without abstract_ref to keep file small) ---
    chunks_for_rag = [
        {k: v for k, v in c.items() if k != 'abstract_ref'}
        for c in all_chunks
    ]
    save_json(chunks_for_rag, DATA_DIR / "paper_chunks.json")

    # --- Build training data (multiple Q&A pairs per chunk) ---
    training_data = []
    abstract_count = 0
    body_count = 0
    paper_qa_count = 0
    medical_qa_count = 0
    for chunk in all_chunks:
        paper_qa_before = paper_qa_count + medical_qa_count
        for example in build_training_examples(chunk):
            # Detect medical Q&A vs paper-centric by checking for condition keywords in instruction
            is_medical = not any(
                marker in example['instruction']
                for marker in ["'", "the paper", "the study", "Summarize", "methods were"]
            )
            if is_medical:
                medical_qa_count += 1
            else:
                paper_qa_count += 1
            training_data.append(example)
        if chunk['is_abstract_ref']:
            abstract_count += 1
        else:
            body_count += 1

    save_json(training_data, DATA_DIR / "papers_training_data.json")

    print(f"\nCollected : {len(papers)} papers")
    print(f"Chunks    : {len(all_chunks)} total")
    print(f"  abstract refs : {abstract_count}")
    print(f"  body chunks   : {body_count}")
    print(f"Training  : {len(training_data)} examples (~{len(training_data)//max(len(all_chunks),1):.1f} per chunk)")
    print(f"  paper-centric Q&A : {paper_qa_count}")
    print(f"  medical Q&A       : {medical_qa_count} (condition-specific, matches evaluation format)")

    return {
        "papers": len(papers),
        "chunks": len(all_chunks),
        "abstract_refs": abstract_count,
        "body_chunks": body_count,
        "training": len(training_data),
    }


def main():
    run()


if __name__ == "__main__":
    main()
