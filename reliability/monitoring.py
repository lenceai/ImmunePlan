"""
Chapter 10: Monitoring, Observability & LLMOps

Implements LLM-native monitoring for the medical AI system:
  - Request/response logging with full traces
  - Quality metrics tracking
  - Cost monitoring and alerts
  - User feedback collection
  - Performance dashboards
  - Three-pillar quality assurance (prevention, detection, correction)

Key principle: "A service can be 'up' and still be failing users."
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from reliability.config import MONITORING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    """Complete trace of a single request through the system."""
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_query: str = ""
    query_type: str = ""
    doctor_type: str = ""

    # Processing steps
    input_safety_check: Optional[Dict] = None
    retrieval_result: Optional[Dict] = None
    tool_calls: List[Dict] = field(default_factory=list)
    prompt_template_used: str = ""

    # Model interaction
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time_seconds: float = 0.0

    # Output
    response: str = ""
    output_safety_check: Optional[Dict] = None
    evaluation_result: Optional[Dict] = None

    # Quality
    groundedness_score: float = 0.0
    overall_quality_score: float = 0.0
    cache_hit: bool = False

    # Cost
    estimated_cost_usd: float = 0.0

    # Feedback
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "user_query": self.user_query[:200],
            "query_type": self.query_type,
            "doctor_type": self.doctor_type,
            "model_used": self.model_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_time_seconds": self.generation_time_seconds,
            "response_length": len(self.response),
            "groundedness_score": self.groundedness_score,
            "overall_quality_score": self.overall_quality_score,
            "cache_hit": self.cache_hit,
            "estimated_cost_usd": self.estimated_cost_usd,
            "safety_issues": (self.output_safety_check or {}).get("issues", []),
            "tool_calls_count": len(self.tool_calls),
            "user_rating": self.user_rating,
        }


class MonitoringService:
    """
    Central monitoring service for tracking system health and quality.

    Three pillars of quality assurance:
    1. Prevention: Input validation, prompt engineering
    2. Detection: Quality metrics, hallucination checks
    3. Correction: Feedback loops, improvement tracking
    """

    def __init__(self):
        self.config = MONITORING_CONFIG
        self.traces: List[RequestTrace] = []
        self.alerts: List[Dict] = []
        self._request_count = 0
        self._error_count = 0
        self._total_cost = 0.0
        self._hourly_costs: Dict[str, float] = defaultdict(float)

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def create_trace(self, query: str, doctor_type: str = "immune") -> RequestTrace:
        """Create a new request trace."""
        self._request_count += 1
        trace = RequestTrace(
            request_id=f"req_{int(time.time() * 1000)}_{self._request_count}",
            user_query=query,
            doctor_type=doctor_type,
        )
        return trace

    def complete_trace(self, trace: RequestTrace):
        """Complete and store a request trace."""
        self.traces.append(trace)
        self._total_cost += trace.estimated_cost_usd

        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        self._hourly_costs[hour_key] += trace.estimated_cost_usd

        self._check_alerts(trace)

        if self.config.enable_request_logging:
            self._log_trace(trace)

    def record_error(self, trace: RequestTrace, error: str):
        """Record an error for a request."""
        self._error_count += 1
        trace.evaluation_result = {"error": error}
        self.complete_trace(trace)

    def record_feedback(self, request_id: str, rating: int, feedback: str = ""):
        """Record user feedback for a request."""
        for trace in reversed(self.traces):
            if trace.request_id == request_id:
                trace.user_rating = rating
                trace.user_feedback = feedback
                break

    def get_dashboard(self) -> Dict:
        """Get monitoring dashboard data."""
        if not self.traces:
            return {"status": "no_data", "total_requests": 0}

        recent = self.traces[-100:]

        latencies = [t.generation_time_seconds for t in recent if t.generation_time_seconds > 0]
        quality_scores = [t.overall_quality_score for t in recent if t.overall_quality_score > 0]
        groundedness = [t.groundedness_score for t in recent if t.groundedness_score > 0]
        ratings = [t.user_rating for t in recent if t.user_rating is not None]
        cache_hits = sum(1 for t in recent if t.cache_hit)

        safety_issues_count = sum(
            len((t.output_safety_check or {}).get("issues", []))
            for t in recent
        )

        query_types = defaultdict(int)
        for t in recent:
            query_types[t.query_type] += 1

        return {
            "status": "healthy" if self._error_count / max(self._request_count, 1) < 0.05 else "degraded",
            "total_requests": self._request_count,
            "recent_requests": len(recent),
            "error_count": self._error_count,
            "error_rate_percent": (self._error_count / max(self._request_count, 1)) * 100,

            "performance": {
                "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0,
                "p95_latency_seconds": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else (max(latencies) if latencies else 0),
                "cache_hit_rate": cache_hits / len(recent) if recent else 0,
            },

            "quality": {
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "avg_groundedness": sum(groundedness) / len(groundedness) if groundedness else 0,
                "safety_issues_count": safety_issues_count,
                "avg_user_rating": sum(ratings) / len(ratings) if ratings else None,
            },

            "cost": {
                "total_cost_usd": self._total_cost,
                "avg_cost_per_request_usd": self._total_cost / max(self._request_count, 1),
                "current_hour_cost_usd": self._hourly_costs.get(datetime.now().strftime("%Y-%m-%d-%H"), 0),
            },

            "query_distribution": dict(query_types),
            "active_alerts": len(self.alerts),
        }

    def _check_alerts(self, trace: RequestTrace):
        """Check if any alert thresholds are exceeded."""
        thresholds = self.config.alert_thresholds

        if trace.generation_time_seconds > thresholds.get("latency_p95_seconds", 15.0):
            self._raise_alert("HIGH_LATENCY", f"Request {trace.request_id} took {trace.generation_time_seconds:.2f}s")

        error_rate = (self._error_count / max(self._request_count, 1)) * 100
        if error_rate > thresholds.get("error_rate_percent", 5.0):
            self._raise_alert("HIGH_ERROR_RATE", f"Error rate at {error_rate:.1f}%")

        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        hourly_cost = self._hourly_costs.get(hour_key, 0)
        if hourly_cost > thresholds.get("cost_per_hour_usd", 5.0):
            self._raise_alert("HIGH_COST", f"Hourly cost at ${hourly_cost:.2f}")

    def _raise_alert(self, alert_type: str, message: str):
        """Raise a monitoring alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "severity": "warning",
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

    def _log_trace(self, trace: RequestTrace):
        """Log a request trace to file."""
        log_file = Path(self.config.log_dir) / f"traces_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log trace: {e}")

    def save_dashboard(self, path: Optional[str] = None):
        """Save dashboard data to file."""
        path = path or str(Path(self.config.log_dir) / "dashboard.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_dashboard(), f, indent=2)

    def get_improvement_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        dashboard = self.get_dashboard()

        if dashboard["status"] == "no_data":
            return ["Insufficient data for recommendations"]

        perf = dashboard["performance"]
        quality = dashboard["quality"]
        cost = dashboard["cost"]

        if perf.get("avg_latency_seconds", 0) > 10:
            recommendations.append(
                "High average latency detected. Consider enabling semantic caching "
                "or optimizing RAG retrieval pipeline."
            )

        if perf.get("cache_hit_rate", 0) < 0.1:
            recommendations.append(
                "Low cache hit rate. Review semantic cache threshold or "
                "increase cache capacity for frequently asked questions."
            )

        if quality.get("avg_groundedness", 1.0) < 0.7:
            recommendations.append(
                "Low groundedness scores. Improve RAG retrieval quality: "
                "better embeddings, metadata filtering, or hybrid retrieval."
            )

        if quality.get("safety_issues_count", 0) > 5:
            recommendations.append(
                "Multiple safety issues detected. Review and strengthen "
                "safety filters and prompt constraints."
            )

        if cost.get("avg_cost_per_request_usd", 0) > 0.03:
            recommendations.append(
                "High per-request cost. Consider semantic caching, "
                "smaller models for simple queries, or multi-model routing."
            )

        avg_rating = quality.get("avg_user_rating")
        if avg_rating is not None and avg_rating < 3.5:
            recommendations.append(
                "Low user satisfaction. Review recent low-rated responses "
                "and identify patterns for prompt/retrieval improvement."
            )

        return recommendations or ["System performing within targets. Continue monitoring."]
