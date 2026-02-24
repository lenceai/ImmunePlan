#!/usr/bin/env python3
"""
Step 07: Build Medical Agent
Book: Chapters 6-7 — Agent with tools, memory, MCP-style interfaces, ReAct loop.

Standalone: python pipeline/07_build_agent.py
Output:     Agent test results printed to console
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import setup_logging, print_step, ensure_directories, DATA_DIR
from pipeline.lib.tools import create_medical_tool_registry
from pipeline.lib.rag import VectorStore, RAGPipeline
from pipeline.lib.safety import SafetyPipeline
from pipeline.lib.monitoring import MonitoringService
from pipeline.lib.agent import MedicalAgent, MultiAgentOrchestrator


def run():
    print_step(7, "BUILD MEDICAL AGENT")
    logger = setup_logging("07_build_agent")
    ensure_directories()

    # --- Tools ---
    print("Registering medical tools...")
    tools = create_medical_tool_registry()
    tool_list = tools.list_tools()
    for t in tool_list:
        print(f"  {t['name']}: {t['description'][:60]}...")

    # --- RAG ---
    print("\nInitializing RAG pipeline...")
    store = VectorStore()
    vs_path = DATA_DIR / "vector_store"
    if vs_path.exists():
        store.load(str(vs_path))
        print(f"  Loaded vector store: {len(store.chunks)} chunks")
    else:
        print("  No vector store found — agent will use tool-only mode")

    rag = RAGPipeline(store)

    # --- Safety & Monitoring ---
    safety = SafetyPipeline()
    monitoring = MonitoringService()

    # --- Agent ---
    print("\nBuilding medical agent...")
    agent = MedicalAgent(
        tool_registry=tools, rag_pipeline=rag,
        safety_pipeline=safety, monitoring=monitoring,
    )

    orchestrator = MultiAgentOrchestrator(monitoring)
    orchestrator.register_agent("immune", agent)

    # --- Test queries ---
    test_queries = [
        "What are the diagnostic criteria for rheumatoid arthritis?",
        "My CRP is 45 mg/L. What does this mean?",
        "What are the side effects of methotrexate?",
        "What do the ACR guidelines recommend for RA treatment?",
    ]

    print("\nTesting agent responses:")
    for query in test_queries:
        print(f"\n  Q: {query}")
        result = orchestrator.route_and_process(query)
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Tools used: {len(result.tool_responses)}")
        print(f"  Steps: {len(result.steps)}")
        print(f"  Response: {result.response[:120]}...")

    dashboard = monitoring.get_dashboard()
    print(f"\nAgent dashboard: {dashboard['total_requests']} requests, status={dashboard['status']}")
    logger.info(f"Agent built and tested: {len(test_queries)} queries processed")
    return {"tools": len(tool_list), "queries_tested": len(test_queries)}


def main():
    run()

if __name__ == "__main__":
    main()
