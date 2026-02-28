"""
Chapters 3 & 4: RAG Pipeline with Vector Search

Implements retrieval-augmented generation for grounding responses in
medical literature. Key reliability features:
  - Chunking with metadata preservation
  - Embedding generation and vector storage
  - Hybrid retrieval (dense + keyword)
  - Metadata filtering (source, section, topic)
  - Retrieval quality scoring
  - Fallback behavior when context is insufficient
"""

import json
import hashlib
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np



logger = logging.getLogger(__name__)

class _RAGConfig:
    vector_db_path = str(Path("./data/vector_store"))
    embedding_model = "BAAI/bge-base-en-v1.5"
    chunk_size = 1500
    chunk_overlap = 200
    top_k = 5
    rerank_top_k = 15           # candidates sent to cross-encoder before final top_k cut
    similarity_threshold = 0.7
    use_hybrid_retrieval = True
    use_reranking = True
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    metadata_filters_enabled = True

RAG_CONFIG = _RAGConfig()


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with metadata."""
    text: str
    source: str
    paper_title: str
    section: str
    score: float
    chunk_id: str
    metadata: Dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.paper_title} - {self.section}] (relevance: {self.score:.2f})"


@dataclass
class RetrievalResult:
    """Complete retrieval result with quality assessment."""
    chunks: List[RetrievedChunk]
    query: str
    retrieval_method: str
    quality_score: float
    sufficient_context: bool
    context_text: str
    citations: List[str]


class VectorStore:
    """FAISS-based vector store with BM25 hybrid search for medical literature."""

    def __init__(self, dimension: int = 768):  # 768 for bge-base-en-v1.5
        self.dimension = dimension
        self.index = None
        self.chunks: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self._encoder = None
        self._reranker = None
        self._bm25 = None          # BM25 index rebuilt on add_chunks
        self._bm25_corpus = []     # tokenised corpus for BM25
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback for similarity search")
            self.index = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(RAG_CONFIG.embedding_model)
                self.dimension = self._encoder.get_sentence_embedding_dimension()
                self._init_index()
            except ImportError:
                logger.warning("sentence-transformers not available")
                self._encoder = "unavailable"
        return self._encoder if self._encoder != "unavailable" else None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        encoder = self._get_encoder()
        if encoder is None:
            return None
        embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def _get_reranker(self):
        if self._reranker is None and RAG_CONFIG.use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RAG_CONFIG.reranker_model)
            except Exception as e:
                logger.warning(f"Reranker unavailable: {e}")
                self._reranker = "unavailable"
        return self._reranker if self._reranker != "unavailable" else None

    def _rebuild_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_corpus = [c.get("text", "").lower().split() for c in self.chunks]
            self._bm25 = BM25Okapi(self._bm25_corpus)
        except ImportError:
            self._bm25 = None

    def add_chunks(self, chunks: List[Dict]):
        texts = [c.get("text", "") for c in chunks]
        embeddings = self.encode(texts)
        if embeddings is None:
            logger.warning("Could not encode chunks, storing without embeddings")
            self.chunks.extend(chunks)
            self._rebuild_bm25()
            return

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(chunk.get("text", "").encode()).hexdigest()[:12]
            chunk["chunk_id"] = chunk_id
            self.chunks.append(chunk)
            self.embeddings.append(embeddings[i])

        if self.index is not None:
            import faiss
            self.index.add(embeddings)

        self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> List[RetrievedChunk]:
        query_embedding = self.encode([query])
        if query_embedding is None or len(self.chunks) == 0:
            return self._keyword_search(query, top_k)

        if self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_embedding, min(top_k * 3, self.index.ntotal))
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                    continue
                candidates.append(RetrievedChunk(
                    text=chunk.get("text", ""),
                    source=chunk.get("source", "unknown"),
                    paper_title=chunk.get("paper_title", "Unknown"),
                    section=chunk.get("section", "body"),
                    score=float(score),
                    chunk_id=chunk.get("chunk_id", ""),
                    metadata=chunk,
                ))
            return candidates[:top_k]
        else:
            return self._numpy_search(query_embedding, top_k, metadata_filter)

    def _numpy_search(self, query_embedding: np.ndarray, top_k: int, metadata_filter: Optional[Dict]) -> List[RetrievedChunk]:
        if not self.embeddings:
            return []
        all_embeddings = np.array(self.embeddings)
        scores = np.dot(all_embeddings, query_embedding.T).flatten()
        indices = np.argsort(scores)[::-1]

        results = []
        for idx in indices:
            chunk = self.chunks[idx]
            if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                continue
            results.append(RetrievedChunk(
                text=chunk.get("text", ""),
                source=chunk.get("source", "unknown"),
                paper_title=chunk.get("paper_title", "Unknown"),
                section=chunk.get("section", "body"),
                score=float(scores[idx]),
                chunk_id=chunk.get("chunk_id", ""),
                metadata=chunk,
            ))
            if len(results) >= top_k:
                break
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """BM25Okapi keyword search. Falls back to naive overlap if rank_bm25 unavailable."""
        if self._bm25 is not None:
            query_tokens = query.lower().split()
            scores = self._bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                RetrievedChunk(
                    text=self.chunks[i].get("text", ""),
                    source=self.chunks[i].get("source", "unknown"),
                    paper_title=self.chunks[i].get("paper_title", "Unknown"),
                    section=self.chunks[i].get("section", "body"),
                    score=float(scores[i]),
                    chunk_id=self.chunks[i].get("chunk_id", ""),
                    metadata=self.chunks[i],
                )
                for i in top_indices if i < len(self.chunks)
            ]

        # naive fallback
        query_terms = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            text_terms = set(chunk.get("text", "").lower().split())
            score = len(query_terms & text_terms) / max(len(query_terms), 1)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                text=c.get("text", ""),
                source=c.get("source", "unknown"),
                paper_title=c.get("paper_title", "Unknown"),
                section=c.get("section", "body"),
                score=s,
                chunk_id=c.get("chunk_id", ""),
                metadata=c,
            )
            for s, c in scored[:top_k]
        ]

    def _matches_filter(self, chunk: Dict, filters: Dict) -> bool:
        for key, value in filters.items():
            if isinstance(value, list):
                if chunk.get(key) not in value:
                    return False
            elif chunk.get(key) != value:
                return False
        return True

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2, default=str)
        if self.embeddings:
            np.save(str(path / "embeddings.npy"), np.array(self.embeddings))
        logger.info(f"Vector store saved: {len(self.chunks)} chunks")

    def load(self, path: str) -> bool:
        path = Path(path)
        chunks_file = path / "chunks.json"
        embeddings_file = path / "embeddings.npy"

        if not chunks_file.exists():
            return False

        with open(chunks_file, "r") as f:
            self.chunks = json.load(f)

        if embeddings_file.exists():
            embeddings = np.load(str(embeddings_file))
            self.embeddings = list(embeddings)
            if self.index is not None:
                self._init_index()
                try:
                    import faiss
                    self.dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(embeddings)
                except ImportError:
                    pass

        self._rebuild_bm25()
        logger.info(f"Vector store loaded: {len(self.chunks)} chunks")
        return True


class RAGPipeline:
    """
    Full RAG pipeline with hybrid retrieval and quality assessment.

    Reliability features:
    - Hybrid retrieval (dense + keyword)
    - Quality scoring of retrieval results
    - Insufficient context detection
    - Metadata filtering
    - Source citation generation
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.config = RAG_CONFIG

    def ingest_training_data(self, data_path: str):
        """Ingest chunked training data into the vector store."""
        path = Path(data_path)
        if not path.exists():
            logger.warning(f"Training data not found: {data_path}")
            return

        with open(path, "r") as f:
            chunks = json.load(f)

        logger.info(f"Ingesting {len(chunks)} chunks into vector store")
        self.vector_store.add_chunks(chunks)
        self.vector_store.save(self.config.vector_db_path)

    # Max tokens to use for RAG context (leave room for prompt + response).
    # 2500 ~ 3 full-length chunks; keeps context informative without blowing the window.
    _MAX_CONTEXT_TOKENS = 2500

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def _truncate_context(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Drop lowest-scoring chunks until context fits within token budget."""
        total = 0
        kept = []
        for chunk in chunks:  # already sorted by score desc
            t = self._approx_tokens(chunk.text)
            if total + t > self._MAX_CONTEXT_TOKENS:
                break
            kept.append(chunk)
            total += t
        return kept if kept else chunks[:1]  # always keep at least one

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
        generate_fn=None,  # optional LLM fn(prompt)->str for HyDE
    ) -> RetrievalResult:
        """Retrieve relevant context for a query, with optional HyDE expansion."""
        top_k = top_k or self.config.top_k

        # HyDE: if LLM available, generate a hypothetical answer and blend embeddings
        hyde_query = query
        if generate_fn is not None:
            try:
                hyde_prompt = (
                    f"Write a concise medical answer (3-4 sentences) to: {query}\n"
                    f"Answer based on clinical knowledge:"
                )
                hypothetical = generate_fn(hyde_prompt)
                if hypothetical and len(hypothetical) > 20:
                    hyde_query = f"{query}\n\n{hypothetical[:400]}"
            except Exception:
                hyde_query = query

        # Fetch more candidates before reranking
        fetch_k = self.config.rerank_top_k if self.config.use_reranking else top_k
        dense_results = self.vector_store.search(hyde_query, top_k=fetch_k, metadata_filter=metadata_filter)

        if self.config.use_hybrid_retrieval:
            keyword_results = self.vector_store._keyword_search(hyde_query, top_k=fetch_k)
            combined = self._merge_results(dense_results, keyword_results, fetch_k)
        else:
            combined = dense_results

        # Cross-encoder reranking
        if self.config.use_reranking and len(combined) > 1:
            reranker = self.vector_store._get_reranker()
            if reranker is not None:
                pairs = [(query, c.text) for c in combined]
                try:
                    rerank_scores = reranker.predict(pairs)
                    for chunk, score in zip(combined, rerank_scores):
                        chunk.score = float(score)
                    combined.sort(key=lambda c: c.score, reverse=True)
                except Exception as e:
                    logger.warning(f"Reranking failed, using RRF order: {e}")
        combined = combined[:top_k]

        # Context window management — drop low-scoring chunks that won't fit
        combined = self._truncate_context(combined)

        quality_score = self._assess_retrieval_quality(query, combined)
        sufficient = quality_score >= self.config.similarity_threshold

        context_parts = []
        citations = []
        for i, chunk in enumerate(combined):
            context_parts.append(
                f"[Source {i+1}: {chunk.paper_title} - {chunk.section}]\n{chunk.text}"
            )
            citations.append(chunk.citation)

        return RetrievalResult(
            chunks=combined,
            query=query,
            retrieval_method="hybrid" if self.config.use_hybrid_retrieval else "dense",
            quality_score=quality_score,
            sufficient_context=sufficient,
            context_text="\n\n---\n\n".join(context_parts),
            citations=citations,
        )

    def _merge_results(
        self,
        dense: List[RetrievedChunk],
        keyword: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Merge dense and keyword results using reciprocal rank fusion."""
        k = 60
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense):
            cid = chunk.chunk_id or chunk.text[:50]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(keyword):
            cid = chunk.chunk_id or chunk.text[:50]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            chunk.score = scores[cid]
            results.append(chunk)
        return results

    def _assess_retrieval_quality(self, query: str, chunks: List[RetrievedChunk]) -> float:
        """
        Assess retrieval quality based on relevance signals.

        Uses sigmoid normalisation on chunk scores so that both cosine-similarity
        values (already 0-1) and cross-encoder logits (unbounded) are mapped to
        a sensible 0-1 range.
        """
        if not chunks:
            return 0.0

        import math
        raw_avg = float(np.mean([c.score for c in chunks]))
        # sigmoid: score=0 → 0.5, score=3 → 0.95, score=-3 → 0.05
        norm_score = 1.0 / (1.0 + math.exp(-raw_avg))

        query_terms = set(query.lower().split())
        term_coverage = 0
        for chunk in chunks:
            chunk_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            term_coverage += overlap / max(len(query_terms), 1)
        avg_coverage = term_coverage / len(chunks)
        source_diversity = len(set(c.source for c in chunks)) / max(len(chunks), 1)

        quality = (0.4 * norm_score) + (0.4 * avg_coverage) + (0.2 * source_diversity)
        return min(max(quality, 0.0), 1.0)
