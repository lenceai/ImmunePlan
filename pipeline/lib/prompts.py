"""
Chapter 2: Structured Prompt Templates for Reliability

Implements prompt engineering as behavior specification, not just wording.
Each prompt template includes:
  1. Role / identity
  2. Task objective
  3. Constraints
  4. Grounding instructions
  5. Output schema/format
  6. Fallback behavior when uncertain
  7. Few-shot examples (where applicable)

Also implements:
  - Self-Consistency: generate multiple outputs and take majority vote
  - Configurable parameter profiles per use case
  - Prompt versioning for A/B testing and rollback
  - Query rephrasing for RAG augmentation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from pipeline.config import RELIABILITY_SPEC


@dataclass
class ParameterProfile:
    """
    Configurable generation parameter profile per use case.

    Book recommendation: "Temperature, top-p, max tokens, stop sequences,
    and penalties should all be treated as configurable parameters in your
    deployment pipeline — not hardcoded values."
    """
    name: str
    temperature: float = 0.3
    top_p: float = 0.85
    max_tokens: int = 1024
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    description: str = ""


PARAMETER_PROFILES = {
    "factual": ParameterProfile(
        name="factual", temperature=0.1, top_p=0.7, max_tokens=512,
        frequency_penalty=0.1, presence_penalty=0.0,
        description="Low variance for factual, deterministic responses",
    ),
    "clinical_support": ParameterProfile(
        name="clinical_support", temperature=0.3, top_p=0.85, max_tokens=1024,
        frequency_penalty=0.1, presence_penalty=0.05,
        description="Moderate temperature for balanced medical responses",
    ),
    "creative": ParameterProfile(
        name="creative", temperature=0.7, top_p=0.9, max_tokens=1024,
        frequency_penalty=0.3, presence_penalty=0.2,
        description="Higher variance for suggestions and diverse responses",
    ),
    "self_consistency": ParameterProfile(
        name="self_consistency", temperature=0.6, top_p=0.9, max_tokens=512,
        description="Moderate temperature for diverse outputs in self-consistency voting",
    ),
}


@dataclass
class PromptVersion:
    """Track prompt versions for A/B testing and rollback."""
    version: str
    template_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    metrics: Dict = field(default_factory=dict)


class PromptRegistry:
    """Version-controlled prompt registry for production deployment."""

    def __init__(self):
        self._versions: Dict[str, List[PromptVersion]] = {}

    def register(self, template_name: str, version: str):
        if template_name not in self._versions:
            self._versions[template_name] = []
        self._versions[template_name].append(
            PromptVersion(version=version, template_name=template_name)
        )

    def get_active_version(self, template_name: str) -> Optional[str]:
        versions = self._versions.get(template_name, [])
        for v in reversed(versions):
            if v.is_active:
                return v.version
        return None

    def record_metrics(self, template_name: str, version: str, metrics: Dict):
        for v in self._versions.get(template_name, []):
            if v.version == version:
                v.metrics.update(metrics)


prompt_registry = PromptRegistry()
prompt_registry.register("medical_qa", "v2.0")
prompt_registry.register("diagnosis", "v2.0")
prompt_registry.register("treatment", "v2.0")


@dataclass
class PromptTemplate:
    """Structured prompt template with reliability controls."""
    role: str
    goal: str
    allowed_sources: str
    disallowed_behaviors: str
    output_format: str
    uncertainty_behavior: str
    examples: Optional[List[Dict[str, str]]] = None

    def render(self, user_query: str, context: str = "", history: str = "") -> str:
        sections = [
            f"## Role\n{self.role}",
            f"## Goal\n{self.goal}",
            f"## Allowed Sources\n{self.allowed_sources}",
            f"## Disallowed Behaviors\n{self.disallowed_behaviors}",
            f"## Output Format\n{self.output_format}",
            f"## When Uncertain\n{self.uncertainty_behavior}",
        ]

        if context:
            sections.append(f"## Retrieved Context\n{context}")

        if history:
            sections.append(f"## Conversation History\n{history}")

        if self.examples:
            example_text = ""
            for i, ex in enumerate(self.examples, 1):
                example_text += f"\n### Example {i}\n"
                example_text += f"Q: {ex['question']}\n"
                example_text += f"A: {ex['answer']}\n"
            sections.append(f"## Examples{example_text}")

        if True:
            sections.append(f"## Medical Disclaimer\n{RELIABILITY_SPEC["medical_disclaimer"]}")

        sections.append(f"## User Question\n{user_query}")

        return "\n\n".join(sections)


MEDICAL_QA_PROMPT = PromptTemplate(
    role=(
        "You are Dr. Immunity, an AI-powered autoimmune disease specialist assistant. "
        "You provide evidence-based educational information about autoimmune conditions "
        "including Rheumatoid Arthritis, Crohn's Disease, Lupus, and related disorders. "
        "You are NOT a physician and cannot provide medical diagnoses or prescriptions."
    ),
    goal=(
        "Answer the user's medical question accurately using ONLY the provided context "
        "and established medical knowledge. Cite sources when available. Provide "
        "structured, clear responses that help patients understand their conditions."
    ),
    allowed_sources=(
        "- Retrieved medical literature and research papers\n"
        "- Established clinical guidelines (ACR/EULAR, AGA, etc.)\n"
        "- Standard medical reference ranges\n"
        "- Peer-reviewed journal findings provided in context"
    ),
    disallowed_behaviors=(
        "- Do NOT fabricate study results, drug names, or dosages\n"
        "- Do NOT provide specific treatment plans or prescriptions\n"
        "- Do NOT make definitive diagnoses\n"
        "- Do NOT dismiss symptoms or discourage seeking professional care\n"
        "- Do NOT speculate beyond what the evidence supports\n"
        "- Do NOT process or store patient-identifiable information"
    ),
    output_format=(
        "Structure your response as:\n"
        "1. **Summary**: Brief direct answer (2-3 sentences)\n"
        "2. **Details**: Detailed explanation with medical reasoning\n"
        "3. **Evidence**: Cite relevant sources from context\n"
        "4. **Confidence**: State your confidence level (High/Medium/Low) and why\n"
        "5. **Next Steps**: Recommend appropriate follow-up actions\n"
        "6. **Disclaimer**: Include medical disclaimer"
    ),
    uncertainty_behavior=(
        "If the provided context does not contain sufficient information to answer "
        "the question accurately:\n"
        "- Explicitly state what you DO know from the context\n"
        "- Clearly identify what you are UNCERTAIN about\n"
        "- Recommend the user consult their healthcare provider\n"
        "- Do NOT fill gaps with fabricated information\n"
        "- Rate your confidence as LOW"
    ),
    examples=[
        {
            "question": "What are the diagnostic criteria for rheumatoid arthritis?",
            "answer": (
                "**Summary**: Rheumatoid arthritis is diagnosed using the 2010 "
                "ACR/EULAR classification criteria, which consider joint involvement, "
                "serology, acute phase reactants, and symptom duration.\n\n"
                "**Details**: The criteria assign points across four domains: "
                "joint involvement (0-5 points), serology including RF and anti-CCP "
                "(0-3 points), acute phase reactants ESR/CRP (0-1 point), and "
                "symptom duration (0-1 point). A score of 6 or more out of 10 "
                "classifies definite RA.\n\n"
                "**Evidence**: Based on 2010 ACR/EULAR Classification Criteria "
                "(Aletaha et al., Arthritis & Rheumatism, 2010).\n\n"
                "**Confidence**: High - well-established diagnostic criteria.\n\n"
                "**Next Steps**: If you suspect RA, consult a rheumatologist for "
                "proper evaluation including blood tests and imaging.\n\n"
                "**Disclaimer**: This is educational information only. "
                "Consult your healthcare provider for diagnosis."
            ),
        }
    ],
)


DIAGNOSIS_PROMPT = PromptTemplate(
    role=(
        "You are an AI clinical decision support assistant specializing in "
        "autoimmune disease differential diagnosis. You help healthcare "
        "professionals by synthesizing clinical findings."
    ),
    goal=(
        "Given the clinical presentation and lab results, provide a structured "
        "differential diagnosis analysis using ONLY the evidence provided. "
        "Rank differentials by likelihood with supporting reasoning."
    ),
    allowed_sources=(
        "- Provided clinical data and lab results\n"
        "- Retrieved medical literature\n"
        "- Standard diagnostic criteria and classification systems"
    ),
    disallowed_behaviors=(
        "- Do NOT make a definitive diagnosis\n"
        "- Do NOT recommend specific medications or dosages\n"
        "- Do NOT ignore or minimize any presented findings\n"
        "- Do NOT fabricate lab values or clinical findings"
    ),
    output_format=(
        "Structure your response as:\n"
        "1. **Key Findings**: Summarize the relevant clinical data\n"
        "2. **Differential Diagnosis**: Ranked list with reasoning\n"
        "3. **Supporting Evidence**: Cite criteria met for each differential\n"
        "4. **Recommended Workup**: Additional tests that would help differentiate\n"
        "5. **Red Flags**: Any urgent findings requiring immediate attention\n"
        "6. **Confidence**: Assessment of diagnostic certainty"
    ),
    uncertainty_behavior=(
        "If the clinical picture is ambiguous:\n"
        "- Present multiple plausible differentials with reasoning\n"
        "- Identify which additional data would narrow the diagnosis\n"
        "- Flag any findings that warrant urgent evaluation\n"
        "- Rate confidence as LOW and explain what is needed"
    ),
)


TREATMENT_QUERY_PROMPT = PromptTemplate(
    role=(
        "You are an AI treatment information assistant specializing in "
        "autoimmune disease management. You provide evidence-based information "
        "about treatment options."
    ),
    goal=(
        "Provide comprehensive, evidence-based information about treatment options "
        "for the queried condition, using retrieved medical literature. "
        "Emphasize that all treatment decisions must be made with a physician."
    ),
    allowed_sources=(
        "- Retrieved clinical guidelines and treatment protocols\n"
        "- Published clinical trial results from context\n"
        "- Standard of care recommendations"
    ),
    disallowed_behaviors=(
        "- Do NOT prescribe or recommend specific drugs for the patient\n"
        "- Do NOT provide dosage recommendations\n"
        "- Do NOT suggest stopping current medications\n"
        "- Do NOT fabricate clinical trial results"
    ),
    output_format=(
        "1. **Treatment Overview**: Summary of current treatment landscape\n"
        "2. **Evidence-Based Options**: List options with supporting evidence\n"
        "3. **Considerations**: Factors that influence treatment selection\n"
        "4. **Monitoring**: What should be monitored during treatment\n"
        "5. **Important Note**: Emphasize physician consultation\n"
        "6. **Sources**: Cite retrieved literature"
    ),
    uncertainty_behavior=(
        "If evidence is limited or conflicting:\n"
        "- State what the current evidence shows\n"
        "- Identify areas of ongoing research\n"
        "- Recommend discussion with a specialist\n"
        "- Note that treatment should be individualized"
    ),
)


SAFETY_CHECK_PROMPT = PromptTemplate(
    role="You are a medical AI safety reviewer.",
    goal=(
        "Review the AI-generated response for safety issues before it reaches "
        "the user. Check for hallucinations, unsupported claims, dangerous "
        "advice, and missing disclaimers."
    ),
    allowed_sources="- The original query, retrieved context, and generated response",
    disallowed_behaviors="- Do NOT modify the response, only evaluate it",
    output_format=(
        "Respond with a JSON object:\n"
        '{"safe": true/false, "issues": ["list of issues"], '
        '"severity": "none/low/medium/high/critical", '
        '"recommendation": "pass/modify/block", '
        '"explanation": "reasoning"}'
    ),
    uncertainty_behavior=(
        "If unsure about safety, err on the side of caution and flag for review."
    ),
)


def select_prompt(query_type: str) -> PromptTemplate:
    """Select the appropriate prompt template based on query type."""
    templates = {
        "medical_qa": MEDICAL_QA_PROMPT,
        "diagnosis": DIAGNOSIS_PROMPT,
        "treatment": TREATMENT_QUERY_PROMPT,
        "safety_check": SAFETY_CHECK_PROMPT,
    }
    return templates.get(query_type, MEDICAL_QA_PROMPT)


def classify_query_type(query: str) -> str:
    """Classify the query type to select the right prompt template."""
    query_lower = query.lower()

    diagnosis_keywords = [
        "diagnos", "differential", "criteria", "classify", "present with",
        "symptoms suggest", "rule out", "workup",
    ]
    treatment_keywords = [
        "treat", "therapy", "medication", "drug", "dose", "protocol",
        "management", "dmard", "biologic", "remission",
    ]

    if any(kw in query_lower for kw in diagnosis_keywords):
        return "diagnosis"
    if any(kw in query_lower for kw in treatment_keywords):
        return "treatment"
    return "medical_qa"


def get_parameter_profile(query_type: str) -> ParameterProfile:
    """Select generation parameters based on query type."""
    mapping = {
        "diagnosis": "factual",
        "treatment": "clinical_support",
        "medical_qa": "clinical_support",
        "safety_check": "factual",
    }
    profile_name = mapping.get(query_type, "clinical_support")
    return PARAMETER_PROFILES.get(profile_name, PARAMETER_PROFILES["clinical_support"])


def rephrase_query_for_retrieval(query: str) -> List[str]:
    """
    Generate multiple query variations for improved RAG retrieval.

    Book concept: "Query rephrasing and augmentation using LLMs to add
    relevant keywords and generate multiple query variations."

    This is a rule-based implementation; with an LLM available,
    the model can generate more sophisticated rephrasings.
    """
    variations = [query]

    medical_expansions = {
        "RA": "rheumatoid arthritis",
        "SLE": "systemic lupus erythematosus",
        "IBD": "inflammatory bowel disease",
        "CD": "Crohn's disease",
        "UC": "ulcerative colitis",
        "MTX": "methotrexate",
        "CRP": "C-reactive protein",
        "ESR": "erythrocyte sedimentation rate",
        "ANA": "antinuclear antibody",
        "DMARD": "disease-modifying antirheumatic drug",
        "anti-CCP": "anti-cyclic citrullinated peptide",
        "DAS28": "disease activity score 28 joints",
    }

    expanded = query
    for abbr, full in medical_expansions.items():
        if abbr in query and full.lower() not in query.lower():
            expanded = query.replace(abbr, f"{abbr} ({full})")
    if expanded != query:
        variations.append(expanded)

    query_lower = query.lower()
    if "diagnos" in query_lower:
        variations.append(f"{query} classification criteria biomarkers")
    elif "treat" in query_lower:
        variations.append(f"{query} clinical guidelines evidence-based")
    elif "symptom" in query_lower:
        variations.append(f"{query} differential diagnosis workup")

    return variations[:3]


class SelfConsistency:
    """
    Self-Consistency: generate multiple outputs and take majority vote.

    Book concept: "Generate multiple outputs for the same input and
    cross-verify results. The majority answer across several generations
    is more likely to be correct."

    For high-stakes medical queries, this increases confidence.
    """

    @staticmethod
    def aggregate_responses(responses: List[str], query: str) -> Dict:
        """
        Aggregate multiple responses using consistency analysis.

        Returns the most consistent response with a consistency score.
        """
        if not responses:
            return {"response": "", "consistency_score": 0.0, "num_responses": 0}

        if len(responses) == 1:
            return {"response": responses[0], "consistency_score": 1.0, "num_responses": 1}

        import re

        key_claims_per_response = []
        for resp in responses:
            claims = set()
            for sentence in re.split(r'[.!?]\s+', resp.lower()):
                words = set(w for w in sentence.split() if len(w) > 3)
                if len(words) >= 3:
                    claims.add(frozenset(list(words)[:8]))
            key_claims_per_response.append(claims)

        consistency_scores = []
        for i, claims_i in enumerate(key_claims_per_response):
            overlaps = []
            for j, claims_j in enumerate(key_claims_per_response):
                if i == j:
                    continue
                if claims_i and claims_j:
                    shared = sum(
                        1 for c_i in claims_i
                        for c_j in claims_j
                        if len(c_i & c_j) >= 3
                    )
                    total = max(len(claims_i), 1)
                    overlaps.append(shared / total)
                else:
                    overlaps.append(0.0)
            consistency_scores.append(sum(overlaps) / max(len(overlaps), 1))

        best_idx = consistency_scores.index(max(consistency_scores)) if consistency_scores else 0
        avg_consistency = sum(consistency_scores) / max(len(consistency_scores), 1)

        return {
            "response": responses[best_idx],
            "consistency_score": min(avg_consistency, 1.0),
            "num_responses": len(responses),
            "selected_index": best_idx,
        }
