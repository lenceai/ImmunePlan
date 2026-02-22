"""
Chapter 1: Reliability Specification & Configuration

Defines the reliability requirements, targets, and constraints for the
ImmunePlan medical AI system BEFORE choosing architecture. This is the
foundation all other layers build upon.

Key principle: "Define your project's reliability target before choosing architecture."
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path


class ReliabilityTier(Enum):
    """Reliability tier determines how strict the system behavior is."""
    INFORMATIONAL = "informational"
    CLINICAL_SUPPORT = "clinical_support"
    SAFETY_CRITICAL = "safety_critical"


class FailureMode(Enum):
    """Unacceptable failure categories for medical AI."""
    HALLUCINATED_DIAGNOSIS = "hallucinated_diagnosis"
    INCORRECT_DOSAGE = "incorrect_dosage"
    MISSED_CONTRAINDICATION = "missed_contraindication"
    PII_LEAKAGE = "pii_leakage"
    UNAUTHORIZED_TREATMENT_ADVICE = "unauthorized_treatment_advice"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    BIAS_IN_RECOMMENDATION = "bias_in_recommendation"


@dataclass
class ReliabilitySpec:
    """
    One-page reliability specification for ImmunePlan.

    Answers the key questions from Chapter 1:
    - What does "reliable" mean for this app?
    - What are unacceptable failures?
    - What must be grounded/cited?
    - What actions are allowed vs forbidden?
    - What latency and cost targets matter?
    """

    project_name: str = "ImmunePlan"
    tier: ReliabilityTier = ReliabilityTier.CLINICAL_SUPPORT

    reliability_definition: str = (
        "Consistently useful, grounded in medical literature, "
        "transparent about uncertainty, safe for patient-facing use, "
        "and improvable through monitoring."
    )

    unacceptable_failures: List[FailureMode] = field(default_factory=lambda: [
        FailureMode.HALLUCINATED_DIAGNOSIS,
        FailureMode.INCORRECT_DOSAGE,
        FailureMode.MISSED_CONTRAINDICATION,
        FailureMode.PII_LEAKAGE,
        FailureMode.UNAUTHORIZED_TREATMENT_ADVICE,
    ])

    grounding_requirements: Dict[str, str] = field(default_factory=lambda: {
        "diagnosis_criteria": "Must cite ACR/EULAR or equivalent criteria",
        "treatment_recommendations": "Must reference published guidelines or studies",
        "drug_information": "Must be sourced from approved formulary data",
        "lab_interpretation": "Must reference standard reference ranges",
    })

    allowed_actions: List[str] = field(default_factory=lambda: [
        "retrieve_medical_literature",
        "lookup_drug_interactions",
        "check_lab_reference_ranges",
        "search_clinical_guidelines",
        "assess_disease_activity_scores",
    ])

    forbidden_actions: List[str] = field(default_factory=lambda: [
        "prescribe_medication",
        "order_tests",
        "modify_patient_records",
        "provide_definitive_diagnosis",
        "override_physician_judgment",
    ])

    latency_targets: Dict[str, float] = field(default_factory=lambda: {
        "simple_query_p95_seconds": 5.0,
        "complex_query_p95_seconds": 15.0,
        "rag_retrieval_p95_seconds": 2.0,
    })

    cost_targets: Dict[str, float] = field(default_factory=lambda: {
        "max_cost_per_query_usd": 0.05,
        "max_monthly_budget_usd": 500.0,
    })

    quality_targets: Dict[str, float] = field(default_factory=lambda: {
        "groundedness_score_min": 0.8,
        "hallucination_rate_max": 0.05,
        "citation_rate_min": 0.9,
        "safety_pass_rate_min": 0.99,
        "user_satisfaction_min": 4.0,
    })


@dataclass
class ModelConfig:
    """Model selection and parameter configuration for reliability."""

    primary_model: str = "local"
    fallback_model: str = "local"

    generation_params: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.3,
        "top_p": 0.85,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.1,
    })

    factual_params: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.1,
        "top_p": 0.7,
        "max_new_tokens": 512,
    })

    creative_params: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 1024,
    })


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    vector_db_path: str = str(Path(os.getenv("DATA_DIR", "./data")) / "vector_store")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_hybrid_retrieval: bool = True
    use_reranking: bool = True
    metadata_filters_enabled: bool = True


@dataclass
class SafetyConfig:
    """Safety and responsible AI configuration."""

    enable_pii_detection: bool = True
    enable_bias_check: bool = True
    enable_safety_filter: bool = True
    enable_medical_disclaimer: bool = True
    enable_source_citation: bool = True

    max_confidence_without_source: float = 0.3
    escalation_threshold: float = 0.4

    pii_patterns: List[str] = field(default_factory=lambda: [
        "ssn", "social_security", "date_of_birth", "dob",
        "address", "phone", "email", "mrn", "medical_record",
        "insurance_id", "credit_card",
    ])

    hipaa_identifiers: List[str] = field(default_factory=lambda: [
        "name", "address", "dates", "phone", "fax", "email",
        "ssn", "mrn", "health_plan", "account_number",
        "certificate", "vehicle_id", "device_id", "url",
        "ip_address", "biometric", "photo", "other_unique",
    ])

    medical_disclaimer: str = (
        "IMPORTANT: This information is for educational and informational "
        "purposes only. It is NOT a substitute for professional medical advice, "
        "diagnosis, or treatment. Always seek the advice of your physician or "
        "other qualified health provider with any questions you may have "
        "regarding a medical condition."
    )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    enable_request_logging: bool = True
    enable_cost_tracking: bool = True
    enable_quality_metrics: bool = True
    enable_latency_tracking: bool = True
    enable_feedback_collection: bool = True

    log_dir: str = str(Path(os.getenv("LOGS_DIR", "./logs")) / "monitoring")
    metrics_retention_days: int = 90

    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latency_p95_seconds": 15.0,
        "error_rate_percent": 5.0,
        "hallucination_rate_percent": 5.0,
        "cost_per_hour_usd": 5.0,
    })


RELIABILITY_SPEC = ReliabilitySpec()
MODEL_CONFIG = ModelConfig()
RAG_CONFIG = RAGConfig()
SAFETY_CONFIG = SafetyConfig()
MONITORING_CONFIG = MonitoringConfig()
