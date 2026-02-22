"""
Chapter 9: Evaluation Framework

Implements comprehensive evaluation for the medical AI system:
  - Hallucination detection (grounding checks)
  - Medical accuracy scoring
  - Response quality metrics (structure, terminology, confidence)
  - ROUGE-based summarization quality
  - LLM-as-judge evaluation
  - Agent trajectory evaluation
  - Cost and performance tracking
"""

import re
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


MEDICAL_TERMS = [
    "diagnosis", "diagnostic", "criteria", "antibody", "antibodies",
    "autoimmune", "lupus", "rheumatoid", "arthritis", "syndrome",
    "sclerosis", "vasculitis", "polymyositis", "sjogren",
    "ANA", "RF", "anti-CCP", "anti-dsDNA", "complement", "biopsy",
    "pathophysiology", "etiology", "prognosis", "treatment", "therapy",
    "inflammation", "immunosuppressive", "corticosteroid", "DMARD",
    "cytokine", "autoantibody", "serological", "clinical",
    "crohn", "inflammatory", "bowel", "biologic", "remission",
    "calprotectin", "endoscopy", "colonoscopy", "fistula",
    "methotrexate", "adalimumab", "infliximab", "JAK", "TNF",
]


@dataclass
class EvaluationResult:
    """Structured evaluation result."""
    query: str
    response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    groundedness_score: float = 0.0
    medical_accuracy_score: float = 0.0
    structure_score: float = 0.0
    safety_score: float = 0.0
    overall_score: float = 0.0

    has_citation: bool = False
    has_disclaimer: bool = False
    has_confidence_level: bool = False
    has_next_steps: bool = False
    has_structured_format: bool = False

    medical_term_count: int = 0
    word_count: int = 0
    latency_seconds: float = 0.0

    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class GroundednessChecker:
    """
    Check if responses are grounded in provided context.

    Detects hallucinations by verifying claims against retrieved sources.
    """

    CLAIM_PATTERNS = [
        r"studies?\s+(?:show|demonstrate|indicate|suggest|reveal)",
        r"according\s+to",
        r"research\s+(?:shows|demonstrates|indicates|suggests)",
        r"\d+%\s+of\s+patients",
        r"(?:first|second|third)-line\s+(?:treatment|therapy)",
        r"(?:recommended|approved)\s+(?:dose|dosage)\s+(?:is|of)\s+\d+",
        r"clinical\s+trial[s]?\s+(?:show|demonstrate)",
    ]

    def check(self, response: str, context: str) -> Tuple[float, List[str]]:
        """Check groundedness of response against context."""
        if not context:
            return 0.5, ["No context provided for grounding check"]

        issues = []
        claims_found = 0
        claims_grounded = 0

        for pattern in self.CLAIM_PATTERNS:
            matches = re.finditer(pattern, response.lower())
            for match in matches:
                claims_found += 1
                start = max(0, match.start() - 50)
                end = min(len(response), match.end() + 100)
                claim_text = response[start:end].lower()

                key_terms = set(re.findall(r'\b\w{4,}\b', claim_text))
                context_lower = context.lower()
                grounded_terms = sum(1 for t in key_terms if t in context_lower)

                if grounded_terms / max(len(key_terms), 1) >= 0.3:
                    claims_grounded += 1
                else:
                    issues.append(f"Potentially ungrounded claim near: ...{claim_text[:80]}...")

        if claims_found == 0:
            return 0.8, []

        score = claims_grounded / claims_found
        return score, issues


class ResponseQualityChecker:
    """Evaluate response quality using heuristic metrics."""

    def evaluate(self, response: str, query: str) -> EvaluationResult:
        result = EvaluationResult(query=query, response=response)
        result.word_count = len(response.split())

        response_lower = response.lower()

        result.has_citation = any(
            marker in response_lower
            for marker in ["source", "reference", "cited", "according to", "study", "evidence"]
        )

        result.has_disclaimer = any(
            marker in response_lower
            for marker in ["disclaimer", "not a substitute", "consult", "healthcare provider",
                           "medical advice", "professional"]
        )

        result.has_confidence_level = any(
            marker in response_lower
            for marker in ["confidence", "high confidence", "medium confidence",
                           "low confidence", "uncertain", "likely"]
        )

        result.has_next_steps = any(
            marker in response_lower
            for marker in ["next step", "recommend", "follow-up", "consult your",
                           "suggest", "further evaluation"]
        )

        result.has_structured_format = bool(
            re.search(r'(?:\*\*[^*]+\*\*|\d+\.\s|\-\s|•\s)', response)
        )

        result.medical_term_count = sum(
            1 for term in MEDICAL_TERMS if term.lower() in response_lower
        )

        structure_points = sum([
            result.has_citation,
            result.has_disclaimer,
            result.has_confidence_level,
            result.has_next_steps,
            result.has_structured_format,
        ])
        result.structure_score = structure_points / 5.0

        result.medical_accuracy_score = min(result.medical_term_count / 10.0, 1.0)

        result.safety_score = 1.0
        if not result.has_disclaimer:
            result.safety_score -= 0.3
            result.issues.append("Missing medical disclaimer")
        if "definitely" in response_lower and "diagnos" in response_lower:
            result.safety_score -= 0.3
            result.issues.append("Appears to make definitive diagnosis")
        if re.search(r'(?:take|prescribe|dose)\s+\d+\s*mg', response_lower):
            result.safety_score -= 0.4
            result.issues.append("Contains specific dosage recommendation")

        result.overall_score = (
            0.3 * result.structure_score
            + 0.3 * result.medical_accuracy_score
            + 0.2 * result.safety_score
            + 0.2 * min(result.word_count / 200, 1.0)
        )

        return result


class PerformanceTracker:
    """Track latency, throughput, and cost metrics."""

    def __init__(self):
        self.metrics: List[Dict] = []

    def record(
        self,
        query: str,
        latency_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "local",
        cache_hit: bool = False,
    ):
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "latency_seconds": latency_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "cache_hit": cache_hit,
            "estimated_cost_usd": self._estimate_cost(input_tokens, output_tokens, model),
        }
        self.metrics.append(metric)

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        costs = {
            "local": {"input": 0.0, "output": 0.0},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
        }
        model_cost = costs.get(model, costs["local"])
        return (input_tokens / 1000 * model_cost["input"] +
                output_tokens / 1000 * model_cost["output"])

    def get_summary(self) -> Dict:
        if not self.metrics:
            return {"total_requests": 0}

        latencies = [m["latency_seconds"] for m in self.metrics]
        costs = [m["estimated_cost_usd"] for m in self.metrics]
        tokens = [m["total_tokens"] for m in self.metrics]

        return {
            "total_requests": len(self.metrics),
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "p95_latency_seconds": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "total_cost_usd": sum(costs),
            "avg_cost_per_request_usd": sum(costs) / len(costs),
            "total_tokens": sum(tokens),
            "cache_hit_rate": sum(1 for m in self.metrics if m["cache_hit"]) / len(self.metrics),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"metrics": self.metrics, "summary": self.get_summary()}, f, indent=2)


class SemanticCache:
    """
    Semantic caching to reduce redundant LLM calls.

    Uses embedding similarity to identify semantically equivalent queries
    and return cached responses.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache: Dict[str, str] = {}
        self.query_embeddings: List[np.ndarray] = []
        self.query_keys: List[str] = []
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                self._encoder = "unavailable"
        return self._encoder if self._encoder != "unavailable" else None

    def get(self, query: str) -> Optional[str]:
        import hashlib
        exact_key = hashlib.md5(query.encode()).hexdigest()
        if exact_key in self.cache:
            return self.cache[exact_key]

        encoder = self._get_encoder()
        if encoder and self.query_embeddings:
            query_emb = encoder.encode([query], normalize_embeddings=True)
            all_embs = np.array(self.query_embeddings)
            similarities = np.dot(all_embs, query_emb.T).flatten()
            best_idx = np.argmax(similarities)
            if similarities[best_idx] >= self.threshold:
                return self.cache.get(self.query_keys[best_idx])
        return None

    def put(self, query: str, response: str):
        import hashlib
        exact_key = hashlib.md5(query.encode()).hexdigest()
        self.cache[exact_key] = response

        encoder = self._get_encoder()
        if encoder:
            emb = encoder.encode([query], normalize_embeddings=True)
            self.query_embeddings.append(emb[0])
            self.query_keys.append(exact_key)


try:
    import numpy as np
except ImportError:
    np = None
