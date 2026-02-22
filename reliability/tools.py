"""
Chapters 6 & 7: Tool Integration & MCP-Style Interfaces

Implements standardized tool interfaces for the medical AI agent:
  - Self-describing tools with schemas
  - Consistent response format (success, error_code, data, retryable)
  - Structured error handling
  - Tool registry for discovery
  - Medical-specific tools (lab lookup, drug interaction, guidelines)

Key principle: "Tool design becomes interface design."
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"


@dataclass
class ToolResponse:
    """
    Standardized tool response following MCP principles.

    Every tool returns this format so the model can distinguish:
    - success vs failure
    - empty results vs errors
    - retryable vs permanent failures
    """
    success: bool
    status: ToolStatus
    data: Any = None
    error_code: Optional[str] = None
    message: str = ""
    retryable: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "status": self.status.value,
            "data": self.data,
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
        }

    def to_context_string(self) -> str:
        if self.success:
            if isinstance(self.data, dict):
                import json
                return f"[{self.tool_name}] {json.dumps(self.data, indent=2)}"
            return f"[{self.tool_name}] {self.data}"
        return f"[{self.tool_name} ERROR] {self.message}"


@dataclass
class ToolOutputValidation:
    """
    Validation rules for tool outputs.

    Book concept: "Validate tool outputs — check that returned data is
    within reasonable ranges before the agent uses the output."
    """
    required_keys: List[str] = field(default_factory=list)
    value_ranges: Dict[str, tuple] = field(default_factory=dict)
    max_response_size: int = 50000

    def validate(self, data: Any) -> tuple:
        """Returns (is_valid, issues)."""
        issues = []
        if data is None:
            return False, ["Tool returned None"]

        if isinstance(data, dict):
            for key in self.required_keys:
                if key not in data:
                    issues.append(f"Missing required key: {key}")

            for key, (low, high) in self.value_ranges.items():
                if key in data:
                    try:
                        val = float(data[key])
                        if val < low or val > high:
                            issues.append(f"Value {key}={val} outside range [{low}, {high}]")
                    except (ValueError, TypeError):
                        pass

        if isinstance(data, str) and len(data) > self.max_response_size:
            issues.append(f"Response too large: {len(data)} chars")

        return len(issues) == 0, issues


@dataclass
class ToolSchema:
    """Schema describing a tool for model understanding."""
    name: str
    description: str
    parameters: Dict[str, Dict]
    required_params: List[str]
    returns: str
    examples: List[Dict] = field(default_factory=list)
    validation: Optional[ToolOutputValidation] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
            "returns": self.returns,
            "examples": self.examples,
        }


class ToolRegistry:
    """Registry of available tools with schemas for model discovery."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema, handler: Callable):
        self._tools[schema.name] = handler
        self._schemas[schema.name] = schema

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        return self._schemas.get(name)

    def list_tools(self) -> List[Dict]:
        return [schema.to_dict() for schema in self._schemas.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResponse:
        handler = self._tools.get(tool_name)
        if handler is None:
            return ToolResponse(
                success=False,
                status=ToolStatus.NOT_FOUND,
                error_code="TOOL_NOT_FOUND",
                message=f"Tool '{tool_name}' is not registered",
                tool_name=tool_name,
            )

        start = time.time()
        try:
            result = handler(**kwargs)
            elapsed = (time.time() - start) * 1000
            if isinstance(result, ToolResponse):
                result.execution_time_ms = elapsed
                result.tool_name = tool_name
                return result
            return ToolResponse(
                success=True,
                status=ToolStatus.SUCCESS,
                data=result,
                tool_name=tool_name,
                execution_time_ms=elapsed,
            )
        except TimeoutError:
            return ToolResponse(
                success=False,
                status=ToolStatus.TIMEOUT,
                error_code="TIMEOUT",
                message=f"Tool '{tool_name}' timed out",
                retryable=True,
                tool_name=tool_name,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return ToolResponse(
                success=False,
                status=ToolStatus.ERROR,
                error_code="EXECUTION_ERROR",
                message=str(e),
                retryable=False,
                tool_name=tool_name,
                execution_time_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# MEDICAL TOOLS
# =============================================================================

LAB_REFERENCE_RANGES = {
    "RF": {"unit": "IU/mL", "normal": "<14", "positive": ">14", "high": ">60",
           "description": "Rheumatoid Factor - autoantibody associated with RA"},
    "anti-CCP": {"unit": "U/mL", "normal": "<20", "positive": ">20", "high": ">60",
                 "description": "Anti-cyclic citrullinated peptide - highly specific for RA"},
    "ANA": {"unit": "titer", "normal": "<1:40", "positive": ">=1:80", "high": ">=1:640",
            "description": "Antinuclear antibody - associated with autoimmune conditions"},
    "anti-dsDNA": {"unit": "IU/mL", "normal": "<30", "positive": ">30", "high": ">200",
                   "description": "Anti-double stranded DNA - specific for SLE"},
    "CRP": {"unit": "mg/L", "normal": "<3", "elevated": "3-10", "high": ">10",
            "description": "C-reactive protein - acute phase inflammation marker"},
    "ESR": {"unit": "mm/h", "normal_male": "<15", "normal_female": "<20", "elevated": ">30",
            "description": "Erythrocyte sedimentation rate - inflammation marker"},
    "fecal_calprotectin": {"unit": "ug/g", "normal": "<50", "borderline": "50-200",
                           "elevated": ">200", "high": ">500",
                           "description": "Fecal calprotectin - intestinal inflammation marker"},
    "complement_C3": {"unit": "mg/dL", "normal": "90-180", "low": "<90",
                      "description": "Complement C3 - consumed in active SLE"},
    "complement_C4": {"unit": "mg/dL", "normal": "16-48", "low": "<16",
                      "description": "Complement C4 - consumed in active SLE"},
}


def lookup_lab_reference(lab_name: str, value: Optional[str] = None) -> ToolResponse:
    """Look up reference ranges for a laboratory test."""
    lab_key = lab_name.lower().replace(" ", "_").replace("-", "_")

    for key, data in LAB_REFERENCE_RANGES.items():
        if lab_key in key.lower().replace("-", "_") or key.lower().replace("-", "_") in lab_key:
            result = {
                "test_name": key,
                "reference_data": data,
            }
            if value:
                result["provided_value"] = value
                result["interpretation"] = _interpret_lab_value(key, value, data)
            return ToolResponse(
                success=True, status=ToolStatus.SUCCESS, data=result,
                message=f"Reference range found for {key}",
            )

    return ToolResponse(
        success=False, status=ToolStatus.NOT_FOUND,
        error_code="LAB_NOT_FOUND",
        message=f"No reference range found for '{lab_name}'. Available tests: {', '.join(LAB_REFERENCE_RANGES.keys())}",
    )


def _interpret_lab_value(test_name: str, value: str, reference: Dict) -> str:
    try:
        numeric = float(value.replace(">", "").replace("<", "").replace("=", "").strip())
    except ValueError:
        return f"Cannot interpret non-numeric value: {value}"

    if "normal" in reference:
        normal_str = reference["normal"]
        try:
            if "<" in normal_str:
                threshold = float(normal_str.replace("<", ""))
                if numeric < threshold:
                    return "Within normal range"
                elif "high" in reference:
                    high_threshold = float(reference["high"].replace(">", ""))
                    if numeric > high_threshold:
                        return "Significantly elevated"
                return "Elevated"
        except ValueError:
            pass
    return "Refer to reference ranges for interpretation"


DISEASE_ACTIVITY_SCORES = {
    "DAS28": {
        "description": "Disease Activity Score 28 joints",
        "condition": "Rheumatoid Arthritis",
        "ranges": {
            "remission": "<2.6",
            "low_activity": "2.6-3.2",
            "moderate_activity": "3.2-5.1",
            "high_activity": ">5.1",
        },
    },
    "CDAI_RA": {
        "description": "Clinical Disease Activity Index",
        "condition": "Rheumatoid Arthritis",
        "ranges": {
            "remission": "<=2.8",
            "low_activity": "2.8-10",
            "moderate_activity": "10-22",
            "high_activity": ">22",
        },
    },
    "CDAI_Crohn": {
        "description": "Crohn's Disease Activity Index",
        "condition": "Crohn's Disease",
        "ranges": {
            "remission": "<150",
            "mild": "150-220",
            "moderate": "220-450",
            "severe": ">450",
        },
    },
}


def assess_disease_activity(score_name: str, value: float) -> ToolResponse:
    """Assess disease activity based on a validated scoring system."""
    score_key = score_name.upper().replace(" ", "_").replace("'", "")

    for key, data in DISEASE_ACTIVITY_SCORES.items():
        if score_key in key.upper() or key.upper() in score_key:
            interpretation = _interpret_activity_score(value, data["ranges"])
            return ToolResponse(
                success=True, status=ToolStatus.SUCCESS,
                data={
                    "score_system": key,
                    "description": data["description"],
                    "condition": data["condition"],
                    "value": value,
                    "interpretation": interpretation,
                    "ranges": data["ranges"],
                },
                message=f"{key} = {value}: {interpretation}",
            )

    return ToolResponse(
        success=False, status=ToolStatus.NOT_FOUND,
        error_code="SCORE_NOT_FOUND",
        message=f"Unknown scoring system: '{score_name}'. Available: {', '.join(DISEASE_ACTIVITY_SCORES.keys())}",
    )


def _interpret_activity_score(value: float, ranges: Dict) -> str:
    for level, range_str in ranges.items():
        if "<" in range_str and "=" not in range_str:
            threshold = float(range_str.replace("<", ""))
            if value < threshold:
                return level
        elif "<=" in range_str:
            threshold = float(range_str.replace("<=", ""))
            if value <= threshold:
                return level
        elif ">" in range_str and "=" not in range_str:
            threshold = float(range_str.replace(">", ""))
            if value > threshold:
                return level
        elif "-" in range_str:
            low, high = range_str.split("-")
            if float(low) <= value <= float(high):
                return level
    return "Unable to classify"


DRUG_INTERACTIONS = {
    "methotrexate": {
        "class": "DMARD",
        "common_interactions": ["NSAIDs (increased toxicity)", "trimethoprim (bone marrow suppression)",
                                "alcohol (hepatotoxicity)"],
        "monitoring": ["CBC every 4-8 weeks", "LFTs every 4-8 weeks", "renal function"],
        "contraindications": ["pregnancy", "severe hepatic impairment", "immunodeficiency"],
    },
    "adalimumab": {
        "class": "Anti-TNF biologic",
        "common_interactions": ["live vaccines (contraindicated)", "other biologics (increased infection risk)"],
        "monitoring": ["TB screening before start", "hepatitis B screening", "CBC periodically"],
        "contraindications": ["active infection", "TB", "heart failure NYHA III-IV"],
    },
    "infliximab": {
        "class": "Anti-TNF biologic",
        "common_interactions": ["live vaccines (contraindicated)", "other biologics"],
        "monitoring": ["infusion reactions", "TB screening", "hepatitis B", "CBC"],
        "contraindications": ["active infection", "TB", "moderate-severe heart failure"],
    },
}


def check_drug_info(drug_name: str) -> ToolResponse:
    """Look up drug interaction and monitoring information."""
    drug_key = drug_name.lower().strip()

    for key, data in DRUG_INTERACTIONS.items():
        if drug_key in key or key in drug_key:
            return ToolResponse(
                success=True, status=ToolStatus.SUCCESS,
                data={"drug": key, **data},
                message=f"Drug information found for {key}",
            )

    return ToolResponse(
        success=False, status=ToolStatus.NOT_FOUND,
        error_code="DRUG_NOT_FOUND",
        message=f"No information found for '{drug_name}'. Available: {', '.join(DRUG_INTERACTIONS.keys())}",
    )


def search_clinical_guidelines(condition: str) -> ToolResponse:
    """Search for relevant clinical guidelines by condition."""
    guidelines = {
        "rheumatoid_arthritis": {
            "guidelines": [
                {"name": "2021 ACR Guideline for Treatment of RA",
                 "key_points": ["Treat-to-target strategy", "MTX as first-line DMARD",
                                "Biologic if MTX inadequate after 3 months"]},
                {"name": "2010 ACR/EULAR Classification Criteria",
                 "key_points": ["Score >= 6/10 for definite RA",
                                "Joint involvement, serology, acute phase, duration"]},
            ],
        },
        "crohns_disease": {
            "guidelines": [
                {"name": "AGA Clinical Practice Guidelines on Management of Crohn's Disease",
                 "key_points": ["Risk stratification guides therapy",
                                "Early biologic for high-risk patients",
                                "Treat-to-target: clinical + endoscopic remission"]},
                {"name": "ECCO Guidelines on Crohn's Disease",
                 "key_points": ["Induction vs maintenance phases",
                                "Anti-TNF or vedolizumab for moderate-severe",
                                "Combination therapy consideration"]},
            ],
        },
        "lupus": {
            "guidelines": [
                {"name": "2019 EULAR/ACR Classification Criteria for SLE",
                 "key_points": ["Entry criterion: ANA >= 1:80",
                                "Additive weighted criteria across domains",
                                "Score >= 10 for SLE classification"]},
            ],
        },
    }

    condition_key = condition.lower().replace(" ", "_").replace("'", "")
    for key, data in guidelines.items():
        if condition_key in key or key in condition_key:
            return ToolResponse(
                success=True, status=ToolStatus.SUCCESS,
                data=data,
                message=f"Clinical guidelines found for {key}",
            )

    return ToolResponse(
        success=False, status=ToolStatus.NOT_FOUND,
        error_code="GUIDELINES_NOT_FOUND",
        message=f"No guidelines found for '{condition}'",
    )


def create_medical_tool_registry() -> ToolRegistry:
    """Create and populate the medical tool registry."""
    registry = ToolRegistry()

    registry.register(
        ToolSchema(
            name="lookup_lab_reference",
            description="Look up reference ranges and interpret values for laboratory tests commonly used in autoimmune disease diagnosis",
            parameters={
                "lab_name": {"type": "string", "description": "Name of the lab test (e.g., 'RF', 'anti-CCP', 'CRP', 'ESR')"},
                "value": {"type": "string", "description": "Optional: patient's lab value to interpret"},
            },
            required_params=["lab_name"],
            returns="Reference ranges, interpretation if value provided",
            examples=[
                {"lab_name": "CRP", "value": "35"},
                {"lab_name": "anti-CCP"},
            ],
        ),
        lookup_lab_reference,
    )

    registry.register(
        ToolSchema(
            name="assess_disease_activity",
            description="Assess disease activity level using validated scoring systems (DAS28, CDAI, etc.)",
            parameters={
                "score_name": {"type": "string", "description": "Name of the scoring system (e.g., 'DAS28', 'CDAI_Crohn')"},
                "value": {"type": "number", "description": "The calculated score value"},
            },
            required_params=["score_name", "value"],
            returns="Activity level interpretation and reference ranges",
            examples=[
                {"score_name": "DAS28", "value": 5.8},
                {"score_name": "CDAI_Crohn", "value": 280},
            ],
        ),
        assess_disease_activity,
    )

    registry.register(
        ToolSchema(
            name="check_drug_info",
            description="Look up drug interaction, monitoring, and contraindication information for autoimmune disease medications",
            parameters={
                "drug_name": {"type": "string", "description": "Name of the medication (e.g., 'methotrexate', 'adalimumab')"},
            },
            required_params=["drug_name"],
            returns="Drug class, interactions, monitoring requirements, contraindications",
        ),
        check_drug_info,
    )

    registry.register(
        ToolSchema(
            name="search_clinical_guidelines",
            description="Search for relevant clinical guidelines and treatment protocols by autoimmune condition",
            parameters={
                "condition": {"type": "string", "description": "Name of the condition (e.g., 'rheumatoid arthritis', 'Crohn's disease')"},
            },
            required_params=["condition"],
            returns="Relevant clinical guidelines with key points",
        ),
        search_clinical_guidelines,
    )

    return registry
