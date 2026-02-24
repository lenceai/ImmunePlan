#!/usr/bin/env python3
"""
Step 10: Deploy API with Monitoring
Book: Chapter 10 — API deployment with monitoring dashboards, alerts, cost tracking.

Standalone: python pipeline/10_deploy.py [--start-api]
Output:     Prints monitoring dashboard, optionally starts API server
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import setup_logging, print_step, ensure_directories, save_json, RESULTS_DIR
from pipeline.lib.monitoring import MonitoringService, FailurePatternDetector
from pipeline.lib.tools import create_medical_tool_registry
from pipeline.lib.rag import VectorStore, RAGPipeline
from pipeline.lib.safety import SafetyPipeline
from pipeline.lib.agent import MedicalAgent, MultiAgentOrchestrator


def run():
    print_step(10, "DEPLOY API WITH MONITORING")
    logger = setup_logging("10_deploy")
    ensure_directories()

    # --- Initialize all components ---
    print("Initializing reliability pipeline...")
    monitoring = MonitoringService()
    tools = create_medical_tool_registry()
    store = VectorStore()
    rag = RAGPipeline(store)
    safety = SafetyPipeline()

    agent = MedicalAgent(
        tool_registry=tools, rag_pipeline=rag,
        safety_pipeline=safety, monitoring=monitoring,
    )

    orchestrator = MultiAgentOrchestrator(monitoring)
    orchestrator.register_agent("immune", agent)

    # --- Run sample queries to populate monitoring ---
    sample_queries = [
        "What are the diagnostic criteria for RA?",
        "What is fecal calprotectin and what do levels mean?",
        "What are the treatment options for Crohn's disease?",
    ]

    print("\nProcessing sample queries for monitoring baseline...")
    for query in sample_queries:
        result = orchestrator.route_and_process(query)
        print(f"  {query[:50]}... → {result.confidence} confidence")

    # --- Dashboard ---
    dashboard = monitoring.get_dashboard()
    print("\n--- Monitoring Dashboard ---")
    print(f"  Status: {dashboard['status']}")
    print(f"  Total requests: {dashboard['total_requests']}")

    perf = dashboard.get("performance", {})
    print(f"  Avg latency: {perf.get('avg_latency_seconds', 0):.2f}s")
    print(f"  Cache hit rate: {perf.get('cache_hit_rate', 0):.1%}")

    quality = dashboard.get("quality", {})
    print(f"  Avg quality: {quality.get('avg_quality_score', 0):.2f}")
    print(f"  Avg groundedness: {quality.get('avg_groundedness', 0):.2f}")

    cost = dashboard.get("cost", {})
    print(f"  Total cost: ${cost.get('total_cost_usd', 0):.4f}")

    # --- Failure pattern detection ---
    print("\n--- Failure Pattern Detection ---")
    patterns = FailurePatternDetector.detect(monitoring)
    if patterns:
        for p in patterns:
            print(f"  [{p['severity']}] {p['pattern']}: {p['description'][:60]}...")
    else:
        print("  No failure patterns detected")

    # --- Recommendations ---
    print("\n--- Recommendations ---")
    recs = monitoring.get_improvement_recommendations()
    for rec in recs:
        print(f"  - {rec[:80]}...")

    # --- Save dashboard ---
    save_json(dashboard, RESULTS_DIR / "dashboard.json")
    print(f"\nDashboard saved to {RESULTS_DIR / 'dashboard.json'}")

    # --- Optionally start API ---
    if "--start-api" in sys.argv:
        print("\nStarting API server...")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent.parent / "api.py")])

    return dashboard


def main():
    run()

if __name__ == "__main__":
    main()
