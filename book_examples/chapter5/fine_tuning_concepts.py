"""
Chapter 5: Fine-Tuning Decision Framework

Demonstrates the decision framework for when to fine-tune vs
use prompting vs RAG, and the fine-tuning data preparation process.

Key insight: "Don't fine-tune because it sounds advanced. Fine-tune when
your problem is behavior consistency at scale and prompting/RAG are plateauing."

Decision framework:
  - Prompting: tasks are general, behavior specified via instructions
  - RAG: facts change, knowledge is external/proprietary
  - Fine-tuning: output style must be consistent, domain patterns hard to describe
  - Hybrid (RAG + fine-tuning): need both live knowledge and domain behavior
"""

import json
from typing import List, Dict


def prepare_medical_training_data() -> List[Dict]:
    """
    Prepare training data for medical fine-tuning.

    Key: include normal cases, edge cases, uncertain cases,
    refusal/escalation cases, and citation examples.
    """
    training_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are Dr. Immunity, an autoimmune specialist AI assistant. Provide evidence-based information. Always cite sources. State confidence level. Include medical disclaimer."},
                {"role": "user", "content": "What are the key biomarkers for diagnosing RA?"},
                {"role": "assistant", "content": "**Summary**: The key biomarkers for RA diagnosis include Rheumatoid Factor (RF) and anti-CCP antibodies.\n\n**Details**: RF is positive in ~70-80% of RA patients but is not specific. Anti-CCP antibodies are highly specific (95-98%) for RA and can be detected years before symptom onset. ESR and CRP indicate inflammation levels.\n\n**Evidence**: Based on 2010 ACR/EULAR Classification Criteria.\n\n**Confidence**: High - well-established biomarkers.\n\n**Next Steps**: Consult a rheumatologist for comprehensive evaluation.\n\n**Disclaimer**: This is educational information only. Consult your healthcare provider."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Dr. Immunity, an autoimmune specialist AI assistant. Provide evidence-based information. Always cite sources. State confidence level. Include medical disclaimer."},
                {"role": "user", "content": "Can you prescribe methotrexate for my RA?"},
                {"role": "assistant", "content": "**Summary**: I cannot prescribe medications. I am an AI assistant that provides educational information only.\n\n**Details**: Methotrexate is indeed a common first-line DMARD for RA, but prescribing requires a licensed physician who can evaluate your complete medical history, current medications, liver and kidney function, and other factors.\n\n**Confidence**: High - clear scope limitation.\n\n**Next Steps**: Please schedule an appointment with your rheumatologist to discuss treatment options including methotrexate.\n\n**Disclaimer**: This is educational information only. Never start, stop, or change medications without consulting your healthcare provider."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Dr. Immunity, an autoimmune specialist AI assistant. Provide evidence-based information. Always cite sources. State confidence level. Include medical disclaimer."},
                {"role": "user", "content": "I have joint pain. Do I have lupus?"},
                {"role": "assistant", "content": "**Summary**: Joint pain alone is not sufficient to diagnose lupus. Many conditions cause joint pain, and a proper diagnosis requires comprehensive evaluation.\n\n**Details**: While joint pain (arthralgia) is common in systemic lupus erythematosus (SLE), it's also seen in RA, osteoarthritis, fibromyalgia, and many other conditions. SLE diagnosis requires meeting specific criteria including ANA positivity and involvement of multiple organ systems.\n\n**Confidence**: LOW - Cannot make diagnostic assessments based on a single symptom.\n\n**Next Steps**: If you're experiencing persistent joint pain, please see your primary care physician who can order appropriate blood tests (ANA, anti-dsDNA, complement levels) and refer you to a rheumatologist if needed.\n\n**Disclaimer**: This is educational information only. I cannot diagnose conditions. Please consult your healthcare provider."}
            ]
        },
    ]

    return training_examples


def validate_training_data(examples: List[Dict]) -> Dict:
    """Validate training data quality for fine-tuning."""
    results = {
        "total_examples": len(examples),
        "has_system_prompt": 0,
        "has_disclaimer": 0,
        "has_confidence": 0,
        "has_citations": 0,
        "has_refusal": 0,
        "avg_response_length": 0,
    }

    total_length = 0
    for ex in examples:
        messages = ex.get("messages", [])
        has_system = any(m["role"] == "system" for m in messages)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        if has_system:
            results["has_system_prompt"] += 1

        for msg in assistant_msgs:
            content = msg["content"].lower()
            total_length += len(content.split())

            if "disclaimer" in content:
                results["has_disclaimer"] += 1
            if "confidence" in content:
                results["has_confidence"] += 1
            if "evidence" in content or "source" in content or "based on" in content:
                results["has_citations"] += 1
            if "cannot" in content or "i am an ai" in content or "not sufficient" in content:
                results["has_refusal"] += 1

    results["avg_response_length"] = total_length / max(len(examples), 1)
    return results


def should_fine_tune(metrics: Dict) -> str:
    """Decision framework: should you fine-tune?"""
    reasons_for = []
    reasons_against = []

    if metrics.get("behavior_consistency_needed", False):
        reasons_for.append("Output style/structure must be highly consistent")
    if metrics.get("domain_patterns_hard_to_describe", False):
        reasons_for.append("Domain patterns are hard to describe in prompts")
    if metrics.get("scale_justifies_cost", False):
        reasons_for.append("Repeated tasks justify training cost")
    if metrics.get("prompting_plateaued", False):
        reasons_for.append("Prompting/RAG improvements have plateaued")

    if not metrics.get("sufficient_training_data", True):
        reasons_against.append("Insufficient training data (need 100+ examples)")
    if metrics.get("facts_change_frequently", False):
        reasons_against.append("Facts change frequently (use RAG instead)")
    if not metrics.get("evaluation_pipeline_ready", True):
        reasons_against.append("No evaluation pipeline to measure improvement")

    if len(reasons_for) >= 2 and len(reasons_against) == 0:
        return "RECOMMENDED: Fine-tuning justified"
    elif len(reasons_against) > len(reasons_for):
        return "NOT RECOMMENDED: Address blockers first"
    else:
        return "CONSIDER: Evaluate hybrid approach (RAG + fine-tuning)"


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 5: Fine-Tuning Decision Framework")
    print("=" * 80)
    print()

    print("--- Preparing Training Data ---")
    examples = prepare_medical_training_data()
    print(f"Created {len(examples)} training examples")

    print("\n--- Validating Training Data ---")
    validation = validate_training_data(examples)
    for key, value in validation.items():
        print(f"  {key}: {value}")

    print("\n--- Fine-Tuning Decision ---")
    decision_metrics = {
        "behavior_consistency_needed": True,
        "domain_patterns_hard_to_describe": True,
        "scale_justifies_cost": True,
        "prompting_plateaued": False,
        "sufficient_training_data": True,
        "facts_change_frequently": True,
        "evaluation_pipeline_ready": True,
    }
    decision = should_fine_tune(decision_metrics)
    print(f"  Decision: {decision}")
    print("  Recommendation: Use hybrid RAG + fine-tuning approach")
