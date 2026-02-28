"""
Chapter 11: Safety, Bias Detection, PII Protection & Responsible AI

Implements defense-in-depth responsible AI architecture:
  - Data layer: Bias detection in training data
  - Safety layer: Runtime content filtering
  - Privacy layer: PII detection and HIPAA compliance
  - Application layer: Transparency and explainability

Key principle: "No single guardrail is enough."
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from pipeline.config import RELIABILITY_SPEC as _RS

logger = logging.getLogger(__name__)

class _SafetyConfig:
    enable_pii_detection = True
    enable_bias_check = True
    enable_safety_filter = True
    enable_medical_disclaimer = True
    enable_source_citation = True
    medical_disclaimer = _RS["medical_disclaimer"]
    max_confidence_without_source = 0.3
    pii_patterns = []
    hipaa_identifiers = []

SAFETY_CONFIG = _SafetyConfig()


class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class PIISeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyCheckResult:
    """Result of a safety check on input or output."""
    safe: bool
    level: SafetyLevel
    issues: List[str] = field(default_factory=list)
    pii_detected: List[Dict] = field(default_factory=list)
    bias_flags: List[str] = field(default_factory=list)
    recommendation: str = "pass"
    explanation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "safe": self.safe,
            "level": self.level.value,
            "issues": self.issues,
            "pii_detected": self.pii_detected,
            "bias_flags": self.bias_flags,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }


class PIIDetector:
    """
    Detect personally identifiable information in text.

    Covers HIPAA identifiers and common PII patterns.
    """

    PII_PATTERNS = {
        "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', PIISeverity.CRITICAL),
        "phone": (r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', PIISeverity.HIGH),
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', PIISeverity.MEDIUM),
        "mrn": (r'\b(?:MRN|Medical Record)[\s#:]*\d{5,}\b', PIISeverity.CRITICAL),
        "dob": (r'\b(?:DOB|date of birth|born)[\s:]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', PIISeverity.HIGH),
        "address": (r'\b\d{1,5}\s+(?:\w+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ct|Court|Ln|Lane)\b', PIISeverity.HIGH),
        "insurance_id": (r'\b(?:insurance|policy)[\s#:]*[A-Z0-9]{5,}\b', PIISeverity.HIGH),
        "credit_card": (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', PIISeverity.CRITICAL),
        "ip_address": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', PIISeverity.LOW),
        "date_specific": (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', PIISeverity.MEDIUM),
    }

    PATIENT_NAME_INDICATORS = [
        r'\bpatient\s+name[\s:]+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        r'\bMr\.\s+[A-Z][a-z]+\b',
        r'\bMs\.\s+[A-Z][a-z]+\b',
        r'\bMrs\.\s+[A-Z][a-z]+\b',
        r'\bDr\.\s+[A-Z][a-z]+\b',
    ]

    def detect(self, text: str) -> List[Dict]:
        """Detect PII in text and return findings."""
        findings = []

        for pii_type, (pattern, severity) in self.PII_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": pii_type,
                    "severity": severity.value,
                    "position": match.start(),
                    "length": len(match.group()),
                    "redacted": self._redact(match.group(), pii_type),
                })

        for pattern in self.PATIENT_NAME_INDICATORS:
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append({
                    "type": "patient_name",
                    "severity": PIISeverity.HIGH.value,
                    "position": match.start(),
                    "length": len(match.group()),
                    "redacted": "[REDACTED_NAME]",
                })

        return findings

    def redact(self, text: str) -> str:
        """Redact all detected PII from text."""
        findings = self.detect(text)
        findings.sort(key=lambda f: f["position"], reverse=True)

        redacted = text
        for finding in findings:
            start = finding["position"]
            end = start + finding["length"]
            redacted = redacted[:start] + finding["redacted"] + redacted[end:]

        return redacted

    def _redact(self, value: str, pii_type: str) -> str:
        redaction_map = {
            "ssn": "[REDACTED_SSN]",
            "phone": "[REDACTED_PHONE]",
            "email": "[REDACTED_EMAIL]",
            "mrn": "[REDACTED_MRN]",
            "dob": "[REDACTED_DOB]",
            "address": "[REDACTED_ADDRESS]",
            "insurance_id": "[REDACTED_INSURANCE]",
            "credit_card": "[REDACTED_CC]",
            "ip_address": "[REDACTED_IP]",
            "date_specific": "[REDACTED_DATE]",
        }
        return redaction_map.get(pii_type, "[REDACTED]")


class ContentSafetyFilter:
    """
    Runtime content safety filter for medical AI.

    Checks for dangerous content in both inputs and outputs.
    """

    DANGEROUS_PATTERNS = [
        (r'\b(?:suicide|self[- ]harm|kill\s+(?:my|your)self)\b', "self_harm", SafetyLevel.BLOCKED),
        (r'\b(?:overdose|take\s+all\s+(?:my|the)\s+(?:pills|medication))\b', "self_harm", SafetyLevel.BLOCKED),
        (r'\b(?:stop\s+(?:taking|all)\s+(?:your|my)\s+(?:medication|treatment))\b', "dangerous_advice", SafetyLevel.UNSAFE),
    ]

    PROMPT_INJECTION_PATTERNS = [
        (r'ignore\s+(?:your\s+)?(?:previous|prior|above|all)\s+instructions?', "prompt_injection"),
        (r'disregard\s+(?:your\s+)?(?:previous|prior|above|all)\s+instructions?', "prompt_injection"),
        (r'you\s+are\s+now\s+(?:a|an|the)\s+\w+', "role_override"),
        (r'(?:act|pretend|roleplay|simulate)\s+(?:as|like)\s+(?:a|an)\s+\w+\s+(?:without|that\s+(?:ignores?|has\s+no))', "role_override"),
        (r'(?:new\s+)?system\s+prompt[\s:]+', "system_override"),
        (r'(?:jailbreak|dan\s+mode|developer\s+mode)', "jailbreak"),
        (r'reveal\s+(?:your\s+)?(?:system\s+prompt|instructions?|training)', "extraction"),
        (r'what\s+(?:are|is)\s+your\s+(?:system\s+prompt|instructions?|initial\s+prompt)', "extraction"),
    ]

    MEDICAL_SAFETY_CHECKS = [
        (r'\byou\s+(?:have|definitely\s+have)\s+(?:cancer|lupus|RA|Crohn)', "definitive_diagnosis"),
        (r'\btake\s+\d+\s*(?:mg|ml|g)\s+(?:of\s+)?\w+\b', "specific_dosage"),
        (r'\b(?:guaranteed|100%|certainly|definitely)\s+(?:cure|heal|fix)\b', "false_guarantee"),
        (r'\bstop\s+(?:seeing|visiting)\s+(?:your\s+)?(?:doctor|physician)\b', "discouraging_care"),
    ]

    @staticmethod
    def _normalize(text: str) -> str:
        """NFKC-normalize to catch Unicode homoglyph bypasses."""
        return unicodedata.normalize("NFKC", text)

    def check_input(self, text: str) -> SafetyCheckResult:
        """Check user input for safety concerns, including prompt injection."""
        normalized = self._normalize(text)
        issues = []
        level = SafetyLevel.SAFE

        for pattern, category, severity in self.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                issues.append(f"Input safety concern: {category}")
                if severity.value == "blocked":
                    level = SafetyLevel.BLOCKED
                elif level != SafetyLevel.BLOCKED:
                    level = severity

        for pattern, category in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                issues.append(f"Prompt injection detected: {category}")
                level = SafetyLevel.BLOCKED

        return SafetyCheckResult(
            safe=level == SafetyLevel.SAFE,
            level=level,
            issues=issues,
            recommendation="block" if level == SafetyLevel.BLOCKED else "pass",
            explanation="Input flagged for safety review" if issues else "Input passes safety checks",
        )

    def check_output(self, response: str) -> SafetyCheckResult:
        """Check AI response for safety concerns."""
        issues = []
        level = SafetyLevel.SAFE

        for pattern, category in self.MEDICAL_SAFETY_CHECKS:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Output safety concern: {category}")
                level = SafetyLevel.CAUTION

        has_disclaimer = any(
            phrase in response.lower()
            for phrase in ["disclaimer", "not a substitute", "consult", "healthcare provider"]
        )
        if not has_disclaimer and len(response) > 200:
            issues.append("Missing medical disclaimer in substantive response")
            if level == SafetyLevel.SAFE:
                level = SafetyLevel.CAUTION

        return SafetyCheckResult(
            safe=level in (SafetyLevel.SAFE, SafetyLevel.CAUTION),
            level=level,
            issues=issues,
            recommendation="modify" if issues else "pass",
            explanation="Response needs modification" if issues else "Response passes safety checks",
        )


class BiasDetector:
    """
    Detect potential bias in AI responses.

    Checks for demographic, treatment, and representation bias.
    """

    DEMOGRAPHIC_BIAS_PATTERNS = [
        (r'\b(?:this disease|condition)\s+(?:only|mainly|primarily)\s+affects?\s+(?:women|men|females|males)\b', "gender_absolutism"),
        (r'\b(?:more common|typical)\s+in\s+(?:white|black|asian|hispanic|caucasian)\b', "racial_generalization"),
        (r'\b(?:older|younger)\s+patients?\s+(?:can\'t|cannot|shouldn\'t)\b', "age_discrimination"),
    ]

    def check(self, text: str) -> List[str]:
        """Check text for potential bias."""
        flags = []

        for pattern, bias_type in self.DEMOGRAPHIC_BIAS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"Potential {bias_type} bias detected")

        return flags


class LLMSafetyClassifier:
    """
    Optional second-layer LLM-based safety check.
    Uses the already-loaded model (or a cross-encoder) to catch adversarial inputs
    that regex patterns miss. Initialise with a loaded model+tokenizer, or leave
    as None to skip (regex layer still applies).
    """

    # Concise safety prompt — model answers "SAFE" or "UNSAFE"
    _PROMPT = (
        "You are a medical AI safety classifier. "
        "Determine if the following user message is safe for a clinical-support chatbot. "
        "Reply with exactly one word: SAFE or UNSAFE.\n\nUser message: {text}\n\nClassification:"
    )

    def __init__(self, model: Any = None, tokenizer: Any = None):
        self._model = model
        self._tokenizer = tokenizer

    def classify(self, text: str) -> Tuple[bool, str]:
        """Returns (is_safe, reason). Fails open (safe=True) if model unavailable."""
        if self._model is None or self._tokenizer is None:
            return True, "llm_classifier_not_loaded"
        try:
            import torch
            prompt = self._PROMPT.format(text=text[:500])
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=5, do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            answer = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().upper()
            is_safe = "UNSAFE" not in answer
            return is_safe, answer
        except Exception as e:
            logger.warning(f"LLM safety classifier error: {e}")
            return True, "error"


class SafetyPipeline:
    """
    Complete safety pipeline implementing defense-in-depth.

    Layers:
    1. PII Detection & Redaction (Privacy layer)
    2. Input Safety Filter (Safety layer)
    3. Output Safety Check (Safety layer)
    4. Bias Detection (Fairness layer)
    5. Disclaimer Enforcement (Application layer)
    """

    def __init__(self, llm_model=None, llm_tokenizer=None):
        self.pii_detector = PIIDetector()
        self.content_filter = ContentSafetyFilter()
        self.bias_detector = BiasDetector()
        self.llm_classifier = LLMSafetyClassifier(llm_model, llm_tokenizer)
        self.config = SAFETY_CONFIG

    def check_input(self, text: str) -> SafetyCheckResult:
        """Run full safety pipeline on user input."""
        pii_findings = []
        if self.config.enable_pii_detection:
            pii_findings = self.pii_detector.detect(text)

        content_result = self.content_filter.check_input(text)
        content_result.pii_detected = pii_findings

        if any(f["severity"] == "critical" for f in pii_findings):
            content_result.issues.append("Critical PII detected in input")
            content_result.recommendation = "redact"

        # LLM second-layer check (only if regex passed to avoid double cost)
        if content_result.level == SafetyLevel.SAFE:
            is_safe, reason = self.llm_classifier.classify(text)
            if not is_safe:
                content_result.safe = False
                content_result.level = SafetyLevel.BLOCKED
                content_result.issues.append(f"LLM safety classifier: UNSAFE ({reason})")
                content_result.recommendation = "block"

        return content_result

    def check_output(self, response: str, context: str = "") -> SafetyCheckResult:
        """Run full safety pipeline on AI output."""
        content_result = self.content_filter.check_output(response)

        if self.config.enable_pii_detection:
            pii_findings = self.pii_detector.detect(response)
            content_result.pii_detected = pii_findings
            if pii_findings:
                content_result.issues.append(f"PII found in output: {len(pii_findings)} instances")
                content_result.recommendation = "redact"

        if self.config.enable_bias_check:
            bias_flags = self.bias_detector.check(response)
            content_result.bias_flags = bias_flags
            if bias_flags:
                content_result.issues.extend(bias_flags)

        return content_result

    def sanitize_input(self, text: str) -> str:
        """Redact PII from user input before processing."""
        if self.config.enable_pii_detection:
            return self.pii_detector.redact(text)
        return text

    def sanitize_output(self, response: str) -> str:
        """Redact PII from AI output and ensure disclaimer."""
        if self.config.enable_pii_detection:
            response = self.pii_detector.redact(response)

        if self.config.enable_medical_disclaimer:
            has_disclaimer = any(
                phrase in response.lower()
                for phrase in ["disclaimer", "not a substitute", "consult your", "healthcare provider"]
            )
            if not has_disclaimer and len(response) > 100:
                response += f"\n\n---\n{self.config.medical_disclaimer}"

        return response

    def get_crisis_response(self) -> str:
        """Return crisis resource information for safety-critical inputs."""
        return (
            "If you or someone you know is in crisis, please contact:\n"
            "- National Suicide Prevention Lifeline: 988 (call or text)\n"
            "- Crisis Text Line: Text HOME to 741741\n"
            "- Emergency Services: 911\n\n"
            "Please reach out to a qualified healthcare professional immediately."
        )
