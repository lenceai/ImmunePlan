"""
End-to-End Reliability Pipeline

Integrates all reliability layers into a single, composable pipeline:

Phase 1 - Reliable Outputs (Ch. 2-5):
  Prompts -> RAG -> Embeddings -> Fine-tuning

Phase 2 - Reliable Agents (Ch. 6-8):
  Tools -> Agent -> Multi-Agent Orchestration

Phase 3 - Reliable Operations (Ch. 9-11):
  Evaluation -> Monitoring -> Safety/Responsible AI

This is the main entry point for the ImmunePlan reliability framework.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

from reliability.config import (
    RELIABILITY_SPEC, MODEL_CONFIG, RAG_CONFIG,
    SAFETY_CONFIG, MONITORING_CONFIG,
)
from reliability.prompts import select_prompt, classify_query_type
from reliability.rag import RAGPipeline, VectorStore
from reliability.tools import create_medical_tool_registry
from reliability.safety import SafetyPipeline
from reliability.monitoring import MonitoringService
from reliability.evaluation import (
    ResponseQualityChecker, GroundednessChecker,
    PerformanceTracker, SemanticCache,
)
from reliability.agent import MedicalAgent, MultiAgentOrchestrator

logger = logging.getLogger(__name__)


class ReliabilityPipeline:
    """
    Complete end-to-end reliability pipeline for ImmunePlan.

    Build order (from the book):
    1. Prompt baseline + tests + function calls (Week 1-2)
    2. RAG + retrieval evaluation + chunking/metadata (Week 2-4)
    3. Embedding/hybrid retrieval + caching + fallback routing (Week 4-6)
    4. Agent/tooling if required (Week 6+)
    5. Observability, cost monitoring, safety/privacy (parallel from day one)
    """

    def __init__(self):
        self._initialized = False
        self.spec = RELIABILITY_SPEC

        # Layer 1: Reliable Outputs
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store)
        self.semantic_cache = SemanticCache()

        # Layer 2: Reliable Agents
        self.tool_registry = create_medical_tool_registry()
        self.safety_pipeline = SafetyPipeline()
        self.monitoring = MonitoringService()

        self.agent = MedicalAgent(
            tool_registry=self.tool_registry,
            rag_pipeline=self.rag_pipeline,
            safety_pipeline=self.safety_pipeline,
            monitoring=self.monitoring,
        )

        self.orchestrator = MultiAgentOrchestrator(self.monitoring)
        self.orchestrator.register_agent("immune", self.agent)

        # Layer 3: Reliable Operations
        self.quality_checker = ResponseQualityChecker()
        self.groundedness_checker = GroundednessChecker()
        self.performance_tracker = PerformanceTracker()

    def initialize(self, data_dir: Optional[str] = None):
        """Initialize the pipeline with training data."""
        if self._initialized:
            return

        data_dir = data_dir or str(Path("./data"))
        chunks_file = Path(data_dir) / "paper_chunks.json"
        vector_store_path = RAG_CONFIG.vector_db_path

        if Path(vector_store_path).exists():
            loaded = self.vector_store.load(vector_store_path)
            if loaded:
                logger.info("Loaded existing vector store")
                self._initialized = True
                return

        if chunks_file.exists():
            logger.info("Ingesting training data into vector store")
            self.rag_pipeline.ingest_training_data(str(chunks_file))
            self._initialized = True
        else:
            logger.warning(
                f"No training data found at {chunks_file}. "
                "RAG will use keyword fallback."
            )
            self._initialized = True

    def set_llm_generator(self, generate_fn):
        """Set the LLM generation function for the agent."""
        self.agent.set_generator(generate_fn)

    def process_query(
        self,
        query: str,
        doctor_type: str = "immune",
        context: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a query through the full reliability pipeline.

        Returns a structured response with metadata about reliability checks.
        """
        if not self._initialized:
            self.initialize()

        # Check semantic cache
        if use_cache:
            cached = self.semantic_cache.get(query)
            if cached:
                logger.info("Semantic cache hit")
                return {
                    "response": cached,
                    "cache_hit": True,
                    "doctor_type": doctor_type,
                    "confidence": "cached",
                    "disclaimer": SAFETY_CONFIG.medical_disclaimer,
                }

        # Process through agent pipeline
        result = self.orchestrator.route_and_process(query, doctor_type, context)

        # Cache the result
        if use_cache and result.confidence in ("high", "medium"):
            self.semantic_cache.put(query, result.response)

        # Track performance
        self.performance_tracker.record(
            query=query,
            latency_seconds=result.processing_time_seconds,
            model="local",
            cache_hit=False,
        )

        return {
            "response": result.response,
            "cache_hit": False,
            "doctor_type": doctor_type,
            "doctor_name": "Dr. Immunity",
            "intent": result.intent.value,
            "confidence": result.confidence,
            "quality_score": result.quality_score,
            "citations": result.citations,
            "safety_check": result.safety_check.to_dict() if result.safety_check else None,
            "tool_calls": [tr.to_dict() for tr in result.tool_responses],
            "retrieval": {
                "chunks_found": len(result.retrieval.chunks) if result.retrieval else 0,
                "quality_score": result.retrieval.quality_score if result.retrieval else 0,
                "sufficient_context": result.retrieval.sufficient_context if result.retrieval else False,
            } if result.retrieval else None,
            "processing_time_seconds": result.processing_time_seconds,
            "steps": [{"type": s.step_type, "content": s.content} for s in result.steps],
            "disclaimer": result.disclaimer or SAFETY_CONFIG.medical_disclaimer,
        }

    def get_dashboard(self) -> Dict:
        """Get the monitoring dashboard."""
        dashboard = self.monitoring.get_dashboard()
        dashboard["recommendations"] = self.monitoring.get_improvement_recommendations()
        dashboard["performance_summary"] = self.performance_tracker.get_summary()
        dashboard["reliability_spec"] = {
            "project": self.spec.project_name,
            "tier": self.spec.tier.value,
            "quality_targets": self.spec.quality_targets,
        }
        return dashboard

    def submit_feedback(self, request_id: str, rating: int, feedback: str = ""):
        """Submit user feedback for a request."""
        self.monitoring.record_feedback(request_id, rating, feedback)

    def get_tool_schemas(self) -> list:
        """Get all registered tool schemas."""
        return self.tool_registry.list_tools()


# Global pipeline instance
_pipeline = None


def get_pipeline() -> ReliabilityPipeline:
    """Get or create the global reliability pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ReliabilityPipeline()
    return _pipeline
