"""
Chapter 11: Responsible AI - Bias, Privacy, and Safety

Demonstrates defense-in-depth responsible AI architecture:
  - Data layer: bias detection in training data
  - Model layer: fairness constraints
  - Safety layer: runtime filtering
  - Privacy layer: PII detection and HIPAA compliance
  - Application layer: transparency and explainability

Key insight: "Responsible AI controls belong in production architecture,
not a policy doc."
"""

import re
from typing import Dict, List, Tuple


def demonstrate_pii_detection():
    """
    PII Detection for HIPAA Compliance.

    HIPAA defines 18 categories of protected health information (PHI).
    This demonstrates detection and redaction of common PHI patterns.
    """
    test_texts = [
        "Patient John Smith, DOB 03/15/1980, SSN 123-45-6789",
        "Send results to john.smith@email.com or call 555-123-4567",
        "MRN: 12345678, Insurance ID: BC123456789",
        "Patient presents with joint pain and morning stiffness",
    ]

    patterns = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "Phone": r'\b\d{3}-\d{3}-\d{4}\b',
        "MRN": r'MRN[\s:]*\d{5,}',
        "DOB": r'DOB\s+\d{2}/\d{2}/\d{4}',
    }

    print("PII Detection Results:")
    for text in test_texts:
        findings = []
        for pii_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                findings.append(pii_type)

        status = "CLEAN" if not findings else f"PII FOUND: {', '.join(findings)}"
        print(f"  [{status}] {text[:60]}...")


def demonstrate_bias_detection():
    """
    Bias Detection in AI Responses.

    Four failure modes from Chapter 11:
    1. Bias / unfair treatment
    2. Safety violations / harmful content
    3. Privacy breaches / data leakage
    4. Opacity / lack of explainability
    """
    test_responses = [
        ("Rheumatoid arthritis primarily affects women, so male patients should consider other diagnoses first.",
         "gender_bias"),
        ("This condition affects all demographics equally, and treatment should be individualized based on disease activity.",
         "unbiased"),
        ("Only white patients typically respond well to this treatment.",
         "racial_bias"),
        ("Treatment response varies by individual. Genetic factors, disease severity, and comorbidities all play a role.",
         "unbiased"),
    ]

    bias_patterns = [
        (r'(?:only|mainly|primarily)\s+(?:affects?|for)\s+(?:women|men|white|black)', "demographic_bias"),
        (r'(?:should not|cannot|won\'t work for)\s+(?:older|younger|female|male)', "exclusion_bias"),
    ]

    print("\nBias Detection Results:")
    for response, expected in test_responses:
        detected = []
        for pattern, bias_type in bias_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                detected.append(bias_type)

        status = "BIAS DETECTED" if detected else "OK"
        print(f"  [{status}] {response[:70]}...")
        if detected:
            print(f"           Types: {', '.join(detected)}")


def defense_in_depth_architecture():
    """
    The five-layer defense-in-depth architecture from Chapter 11.
    """
    layers = {
        "1. Data Layer": {
            "purpose": "Prevent bias entering from skewed data",
            "controls": ["Training data auditing", "Balanced representation checks",
                         "Bias metric computation on datasets"],
        },
        "2. Model Layer": {
            "purpose": "Fairness constraints during training",
            "controls": ["Constitutional AI principles", "RLHF/RLAIF alignment",
                         "Behavior policy encoding"],
        },
        "3. Safety Layer": {
            "purpose": "Runtime filtering and safeguards",
            "controls": ["Content safety filters", "Dangerous advice detection",
                         "Medical claim verification"],
        },
        "4. Privacy Layer": {
            "purpose": "PII detection and regulatory compliance",
            "controls": ["HIPAA identifier detection", "PII redaction",
                         "Data retention policies", "Right to deletion"],
        },
        "5. Application Layer": {
            "purpose": "Transparency and explainability",
            "controls": ["Source citations", "Confidence indicators",
                         "Medical disclaimers", "Audit logging"],
        },
    }

    print("\nDefense-in-Depth Architecture:")
    print("=" * 60)
    for layer_name, details in layers.items():
        print(f"\n  {layer_name}")
        print(f"    Purpose: {details['purpose']}")
        for control in details['controls']:
            print(f"    - {control}")


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 11: Responsible AI - Bias, Privacy, and Safety")
    print("=" * 80)
    print()

    demonstrate_pii_detection()
    demonstrate_bias_detection()
    defense_in_depth_architecture()

    print("\n\nSix Principles of Reliable AI:")
    principles = [
        "Reliability is a systems property, not a model property",
        "Grounding beats confidence (RAG/tools > 'smart-sounding' answers)",
        "Retrieval quality determines RAG quality",
        "Fine-tuning is for behavior consistency, not live facts",
        "Agents multiply capability and failure modes together",
        "Responsible AI controls belong in production architecture",
    ]
    for i, p in enumerate(principles, 1):
        print(f"  {i}. {p}")
