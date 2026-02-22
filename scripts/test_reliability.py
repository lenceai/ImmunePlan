#!/usr/bin/env python3
"""
Reliability Framework Test Suite

Tests all components of the reliability framework end-to-end:
  - Chapter 1: Reliability configuration
  - Chapter 2: Prompt templates
  - Chapters 3-4: RAG pipeline
  - Chapters 6-7: Tools and agents
  - Chapter 8: Multi-agent orchestration
  - Chapter 9: Evaluation metrics
  - Chapter 10: Monitoring
  - Chapter 11: Safety and PII detection
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test reliability configuration (Ch. 1)."""
    print("\n--- Chapter 1: Reliability Configuration ---")
    from reliability.config import RELIABILITY_SPEC, SAFETY_CONFIG, MONITORING_CONFIG

    assert RELIABILITY_SPEC.project_name == "ImmunePlan"
    assert RELIABILITY_SPEC.tier.value == "clinical_support"
    assert len(RELIABILITY_SPEC.unacceptable_failures) > 0
    assert len(RELIABILITY_SPEC.allowed_actions) > 0
    assert len(RELIABILITY_SPEC.forbidden_actions) > 0
    assert RELIABILITY_SPEC.quality_targets["hallucination_rate_max"] <= 0.1

    print(f"  Project: {RELIABILITY_SPEC.project_name}")
    print(f"  Tier: {RELIABILITY_SPEC.tier.value}")
    print(f"  Unacceptable failures: {len(RELIABILITY_SPEC.unacceptable_failures)}")
    print(f"  Quality targets: {json.dumps(RELIABILITY_SPEC.quality_targets, indent=4)}")
    print("  PASSED")


def test_prompts():
    """Test structured prompt templates (Ch. 2)."""
    print("\n--- Chapter 2: Prompt Templates ---")
    from reliability.prompts import (
        MEDICAL_QA_PROMPT, DIAGNOSIS_PROMPT, TREATMENT_QUERY_PROMPT,
        select_prompt, classify_query_type,
    )

    assert MEDICAL_QA_PROMPT.role
    assert MEDICAL_QA_PROMPT.goal
    assert MEDICAL_QA_PROMPT.uncertainty_behavior
    assert MEDICAL_QA_PROMPT.examples

    rendered = MEDICAL_QA_PROMPT.render(
        "What are the diagnostic criteria for RA?",
        context="ACR/EULAR 2010 criteria...",
    )
    assert "Role" in rendered
    assert "Goal" in rendered
    assert "When Uncertain" in rendered
    assert "diagnostic criteria for RA" in rendered
    print(f"  Medical QA prompt rendered: {len(rendered)} chars")

    assert classify_query_type("diagnose lupus") == "diagnosis"
    assert classify_query_type("treatment for Crohn's") == "treatment"
    assert classify_query_type("what is autoimmune disease") == "medical_qa"
    print("  Query type classification: PASSED")

    for qt in ["medical_qa", "diagnosis", "treatment"]:
        template = select_prompt(qt)
        assert template is not None
    print("  Prompt selection: PASSED")
    print("  PASSED")


def test_rag():
    """Test RAG pipeline (Ch. 3-4)."""
    print("\n--- Chapters 3-4: RAG Pipeline ---")
    from reliability.rag import VectorStore, RAGPipeline

    store = VectorStore()

    test_chunks = [
        {
            "text": "Rheumatoid arthritis is diagnosed using the 2010 ACR/EULAR classification criteria. Joint involvement, serology, acute phase reactants, and symptom duration are considered.",
            "source": "arXiv",
            "paper_title": "RA Diagnostic Criteria Review",
            "section": "abstract",
        },
        {
            "text": "Anti-CCP antibodies are highly specific for rheumatoid arthritis with specificity of 95-98%. Positive anti-CCP with elevated CRP strongly suggests RA.",
            "source": "PubMed",
            "paper_title": "Biomarkers in RA",
            "section": "results",
        },
        {
            "text": "Crohn's disease is characterized by transmural inflammation that can affect any part of the gastrointestinal tract. Fecal calprotectin above 200 ug/g suggests active inflammation.",
            "source": "arXiv",
            "paper_title": "IBD Biomarkers Review",
            "section": "body",
        },
    ]

    store.add_chunks(test_chunks)
    assert len(store.chunks) == 3
    print(f"  Added {len(store.chunks)} test chunks to vector store")

    pipeline = RAGPipeline(store)
    result = pipeline.retrieve("diagnostic criteria for rheumatoid arthritis")

    assert len(result.chunks) > 0
    assert result.quality_score >= 0
    assert isinstance(result.context_text, str)
    print(f"  Retrieved {len(result.chunks)} chunks (quality: {result.quality_score:.2f})")
    print(f"  Sufficient context: {result.sufficient_context}")
    print(f"  Method: {result.retrieval_method}")
    print("  PASSED")


def test_tools():
    """Test tool registry and medical tools (Ch. 6-7)."""
    print("\n--- Chapters 6-7: Tools & MCP Interface ---")
    from reliability.tools import create_medical_tool_registry

    registry = create_medical_tool_registry()
    tools = registry.list_tools()
    assert len(tools) >= 4
    print(f"  Registered tools: {len(tools)}")
    for tool in tools:
        print(f"    - {tool['name']}: {tool['description'][:60]}...")

    result = registry.execute("lookup_lab_reference", lab_name="CRP", value="35")
    assert result.success
    assert result.data is not None
    print(f"  Lab lookup (CRP=35): {result.message}")

    result = registry.execute("assess_disease_activity", score_name="DAS28", value=5.8)
    assert result.success
    print(f"  Disease activity (DAS28=5.8): {result.data['interpretation']}")

    result = registry.execute("check_drug_info", drug_name="methotrexate")
    assert result.success
    assert "class" in result.data
    print(f"  Drug info (methotrexate): class={result.data['class']}")

    result = registry.execute("search_clinical_guidelines", condition="rheumatoid_arthritis")
    assert result.success
    print(f"  Guidelines (RA): {len(result.data['guidelines'])} found")

    result = registry.execute("nonexistent_tool")
    assert not result.success
    assert result.status.value == "not_found"
    print("  Nonexistent tool: correctly returned not_found")

    for tool in tools:
        schema = tool
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
    print("  All tool schemas valid")
    print("  PASSED")


def test_safety():
    """Test safety pipeline (Ch. 11)."""
    print("\n--- Chapter 11: Safety & Responsible AI ---")
    from reliability.safety import SafetyPipeline, PIIDetector

    pipeline = SafetyPipeline()

    safe_result = pipeline.check_input("What are the symptoms of RA?")
    assert safe_result.safe
    print(f"  Safe input check: {safe_result.level.value}")

    pii_detector = PIIDetector()
    pii_text = "Patient John Smith, SSN 123-45-6789, email john@example.com"
    findings = pii_detector.detect(pii_text)
    assert len(findings) >= 2
    print(f"  PII detection: found {len(findings)} PII instances")

    redacted = pii_detector.redact(pii_text)
    assert "123-45-6789" not in redacted
    assert "john@example.com" not in redacted
    print(f"  PII redaction: {redacted}")

    unsafe_response = "You definitely have rheumatoid arthritis. Take 25mg methotrexate weekly."
    output_check = pipeline.check_output(unsafe_response)
    assert len(output_check.issues) > 0
    print(f"  Unsafe output detection: {len(output_check.issues)} issues found")
    for issue in output_check.issues:
        print(f"    - {issue}")

    long_response = (
        "Rheumatoid arthritis is a chronic inflammatory disorder that primarily affects the joints. "
        "It occurs when the immune system mistakenly attacks the body's own tissues. "
        "The condition can cause joint damage, pain, and swelling throughout the body."
    )
    safe_response = pipeline.sanitize_output(long_response)
    assert "disclaimer" in safe_response.lower() or "substitute" in safe_response.lower() or "medical advice" in safe_response.lower()
    print("  Disclaimer enforcement: PASSED")

    from reliability.safety import BiasDetector
    bias_detector = BiasDetector()
    biased_text = "This disease only affects women and is more common in white patients"
    bias_flags = bias_detector.check(biased_text)
    assert len(bias_flags) > 0
    print(f"  Bias detection: {len(bias_flags)} flags")
    print("  PASSED")


def test_evaluation():
    """Test evaluation framework (Ch. 9)."""
    print("\n--- Chapter 9: Evaluation Framework ---")
    from reliability.evaluation import ResponseQualityChecker, GroundednessChecker, PerformanceTracker

    checker = ResponseQualityChecker()
    good_response = (
        "**Summary**: Rheumatoid arthritis is diagnosed using the ACR/EULAR criteria.\n\n"
        "**Details**: The 2010 classification criteria consider joint involvement, "
        "serology (RF, anti-CCP), acute phase reactants (CRP, ESR), and symptom duration.\n\n"
        "**Evidence**: Based on Aletaha et al., 2010 ACR/EULAR classification criteria.\n\n"
        "**Confidence**: High - well-established diagnostic criteria.\n\n"
        "**Next Steps**: Consult a rheumatologist for evaluation.\n\n"
        "**Disclaimer**: This is not a substitute for professional medical advice."
    )

    result = checker.evaluate(good_response, "diagnostic criteria for RA")
    assert result.has_citation
    assert result.has_disclaimer
    assert result.has_confidence_level
    assert result.has_next_steps
    assert result.has_structured_format
    assert result.overall_score > 0.5
    print(f"  Quality score: {result.overall_score:.2f}")
    print(f"  Structure: {result.structure_score:.2f}")
    print(f"  Medical terms: {result.medical_term_count}")
    print(f"  Safety score: {result.safety_score:.2f}")

    groundedness = GroundednessChecker()
    context = "Rheumatoid arthritis ACR/EULAR criteria classification diagnosis serology"
    score, issues = groundedness.check(good_response, context)
    print(f"  Groundedness: {score:.2f} ({len(issues)} issues)")

    tracker = PerformanceTracker()
    tracker.record("test query", 1.5, 100, 200, "local")
    tracker.record("test query 2", 2.0, 150, 250, "local")
    summary = tracker.get_summary()
    assert summary["total_requests"] == 2
    print(f"  Performance: {summary['avg_latency_seconds']:.2f}s avg latency")
    print("  PASSED")


def test_monitoring():
    """Test monitoring service (Ch. 10)."""
    print("\n--- Chapter 10: Monitoring & LLMOps ---")
    from reliability.monitoring import MonitoringService

    monitoring = MonitoringService()

    trace = monitoring.create_trace("What is RA?", "immune")
    trace.generation_time_seconds = 1.5
    trace.groundedness_score = 0.85
    trace.overall_quality_score = 0.72
    trace.response = "Rheumatoid arthritis is..."
    monitoring.complete_trace(trace)

    trace2 = monitoring.create_trace("Crohn's treatment", "immune")
    trace2.generation_time_seconds = 2.1
    trace2.groundedness_score = 0.9
    trace2.overall_quality_score = 0.8
    trace2.response = "Crohn's disease treatment..."
    monitoring.complete_trace(trace2)

    dashboard = monitoring.get_dashboard()
    assert dashboard["total_requests"] == 2
    assert dashboard["status"] == "healthy"
    print(f"  Dashboard status: {dashboard['status']}")
    print(f"  Total requests: {dashboard['total_requests']}")
    print(f"  Avg latency: {dashboard['performance']['avg_latency_seconds']:.2f}s")
    print(f"  Avg quality: {dashboard['quality']['avg_quality_score']:.2f}")

    recommendations = monitoring.get_improvement_recommendations()
    print(f"  Recommendations: {len(recommendations)}")
    for rec in recommendations[:2]:
        print(f"    - {rec[:80]}...")
    print("  PASSED")


def test_agent():
    """Test agent framework (Ch. 6, 8)."""
    print("\n--- Chapters 6 & 8: Agent & Multi-Agent ---")
    from reliability.agent import MedicalAgent, MultiAgentOrchestrator, IntentRouter, AgentIntent
    from reliability.tools import create_medical_tool_registry
    from reliability.rag import RAGPipeline, VectorStore
    from reliability.safety import SafetyPipeline
    from reliability.monitoring import MonitoringService

    router = IntentRouter()
    assert router.classify("diagnose lupus") == AgentIntent.DIAGNOSIS
    assert router.classify("methotrexate side effects") == AgentIntent.DRUG_INFO
    assert router.classify("CRP test result 35 mg/L") == AgentIntent.LAB_INTERPRETATION
    assert router.classify("ACR guidelines for RA") == AgentIntent.GUIDELINES
    print("  Intent routing: PASSED")

    store = VectorStore()
    store.add_chunks([{
        "text": "Rheumatoid arthritis diagnostic criteria include joint involvement and serology.",
        "source": "test", "paper_title": "Test Paper", "section": "body",
    }])

    agent = MedicalAgent(
        tool_registry=create_medical_tool_registry(),
        rag_pipeline=RAGPipeline(store),
        safety_pipeline=SafetyPipeline(),
        monitoring=MonitoringService(),
    )

    result = agent.process("What are the diagnostic criteria for rheumatoid arthritis?")
    assert result.response
    assert result.intent == AgentIntent.MEDICAL_QA or result.intent == AgentIntent.DIAGNOSIS
    assert len(result.steps) > 0
    print(f"  Agent response: {len(result.response)} chars")
    print(f"  Intent: {result.intent.value}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Steps: {len(result.steps)}")
    for step in result.steps:
        print(f"    [{step.step_type}] {step.content[:60]}...")

    drug_result = agent.process("What are the side effects of methotrexate?")
    assert drug_result.intent == AgentIntent.DRUG_INFO
    assert len(drug_result.tool_responses) > 0
    print(f"  Drug query: tool calls={len(drug_result.tool_responses)}")

    orchestrator = MultiAgentOrchestrator(MonitoringService())
    orchestrator.register_agent("immune", agent)
    orch_result = orchestrator.route_and_process("What is lupus?")
    assert orch_result.response
    print(f"  Orchestrator response: {len(orch_result.response)} chars")
    print("  PASSED")


def test_pipeline():
    """Test end-to-end pipeline."""
    print("\n--- End-to-End Pipeline ---")
    from reliability.pipeline import ReliabilityPipeline

    pipeline = ReliabilityPipeline()
    pipeline.initialize()

    result = pipeline.process_query("What are the diagnostic criteria for rheumatoid arthritis?")
    assert "response" in result
    assert "confidence" in result
    assert "disclaimer" in result
    print(f"  Response: {result['response'][:100]}...")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Cache hit: {result['cache_hit']}")

    result2 = pipeline.process_query("What are the diagnostic criteria for rheumatoid arthritis?")
    print(f"  Second query cache hit: {result2['cache_hit']}")

    dashboard = pipeline.get_dashboard()
    assert "status" in dashboard
    assert "reliability_spec" in dashboard
    print(f"  Dashboard status: {dashboard['status']}")

    tools = pipeline.get_tool_schemas()
    assert len(tools) >= 4
    print(f"  Available tools: {len(tools)}")
    print("  PASSED")


def main():
    print("=" * 70)
    print("IMMUNEPLAN RELIABILITY FRAMEWORK - TEST SUITE")
    print("Building Reliable AI Systems - All Chapters")
    print("=" * 70)

    tests = [
        ("Configuration (Ch. 1)", test_config),
        ("Prompts (Ch. 2)", test_prompts),
        ("RAG Pipeline (Ch. 3-4)", test_rag),
        ("Tools & MCP (Ch. 6-7)", test_tools),
        ("Safety & Responsible AI (Ch. 11)", test_safety),
        ("Evaluation (Ch. 9)", test_evaluation),
        ("Monitoring (Ch. 10)", test_monitoring),
        ("Agent & Multi-Agent (Ch. 6, 8)", test_agent),
        ("End-to-End Pipeline", test_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
