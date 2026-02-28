"""
Chapters 6 & 8: Agent Framework with Memory & Multi-Agent Orchestration

Implements:
  - Agent base class with tool usage and memory
  - ReAct-style reasoning loop (reason -> act -> observe)
  - Memory management (session + persistent)
  - Intent-aware routing
  - Multi-agent coordination with shared state
  - Specialist agents (medical QA, diagnosis, treatment)
  - Workflow-level error handling

Key principles:
  - "Agents need stronger controls than chatbots"
  - "Multi-agent reliability depends on coordination and state management"
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from pipeline.lib.tools import ToolRegistry, ToolResponse
from pipeline.lib.rag import RAGPipeline, RetrievalResult
from pipeline.lib.prompts import select_prompt, classify_query_type, PromptTemplate
from pipeline.lib.safety import SafetyPipeline, SafetyCheckResult
from pipeline.lib.evaluation import ResponseQualityChecker, GroundednessChecker
from pipeline.lib.monitoring import MonitoringService, RequestTrace

logger = logging.getLogger(__name__)


class AgentIntent(Enum):
    MEDICAL_QA = "medical_qa"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    LAB_INTERPRETATION = "lab_interpretation"
    DRUG_INFO = "drug_info"
    GUIDELINES = "guidelines"
    GENERAL = "general"
    CRISIS = "crisis"


@dataclass
class ConversationMemory:
    """
    Hierarchical session memory with three layers:
      1. recent_messages  — last 5 full turns (verbatim)
      2. summary          — compressed older context (LLM-generated or extractive)
      3. extracted_facts  — persistent structured facts (conditions, meds, labs)
    """
    recent_messages: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""                          # compressed older history
    extracted_facts: Dict[str, str] = field(default_factory=dict)
    tool_results: List[Dict] = field(default_factory=list)
    session_id: str = ""
    _full_history: List[Dict[str, str]] = field(default_factory=list)  # archive
    max_recent: int = 5

    # compat alias
    @property
    def messages(self):
        return self.recent_messages

    def add_message(self, role: str, content: str):
        entry = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self._full_history.append(entry)
        self.recent_messages.append(entry)
        # Roll oldest recent into summary when buffer fills
        if len(self.recent_messages) > self.max_recent * 2:
            evicted = self.recent_messages[:-self.max_recent]
            self.recent_messages = self.recent_messages[-self.max_recent:]
            self._compress_into_summary(evicted)
        # Auto-extract facts from user messages
        if role == "user":
            self._extract_facts(content)

    def _compress_into_summary(self, evicted: List[Dict]):
        """Extractive compression: keep first sentence of each evicted turn."""
        lines = []
        for m in evicted:
            text = m["content"]
            first_sentence = text.split(".")[0][:150]
            lines.append(f"{m['role']}: {first_sentence}")
        new_chunk = " | ".join(lines)
        if self.summary:
            # Keep summary bounded at 1000 chars
            combined = self.summary + " … " + new_chunk
            self.summary = combined[-1000:]
        else:
            self.summary = new_chunk

    def _extract_facts(self, text: str):
        """Simple regex-based fact extraction from user messages."""
        import re
        # Conditions
        conditions = ["rheumatoid arthritis", "crohn", "lupus", "sjogren", "vasculitis",
                      "ankylosing spondylitis", "psoriatic arthritis", "IBD", "UC"]
        for c in conditions:
            if c.lower() in text.lower():
                self.extracted_facts["condition"] = c
        # Medications mentioned
        from pipeline.lib.tools import DRUG_INTERACTIONS
        for drug in DRUG_INTERACTIONS:
            if drug.lower() in text.lower():
                meds = self.extracted_facts.get("medications", "")
                if drug not in meds:
                    self.extracted_facts["medications"] = (meds + ", " + drug).strip(", ")
        # Lab values: "CRP 45" or "DAS28 = 5.8"
        lab_match = re.search(r'\b(CRP|ESR|RF|anti-CCP|DAS28|CDAI|calprotectin)\s*[=:is]?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if lab_match:
            self.extracted_facts[f"lab_{lab_match.group(1).lower()}"] = lab_match.group(2)

    def add_tool_result(self, tool_name: str, result: Dict):
        self.tool_results.append({"tool": tool_name, "result": result,
                                  "timestamp": datetime.now().isoformat()})

    def extract_fact(self, key: str, value: str):
        self.extracted_facts[key] = value

    def get_context_summary(self) -> str:
        parts = []
        if self.extracted_facts:
            facts = "; ".join(f"{k}={v}" for k, v in self.extracted_facts.items())
            parts.append(f"[Patient context: {facts}]")
        if self.summary:
            parts.append(f"[Earlier conversation: {self.summary}]")
        for msg in self.recent_messages[-4:]:
            parts.append(f"{msg['role']}: {msg['content'][:300]}")
        return "\n".join(parts) if parts else ""

    def clear(self):
        self.recent_messages.clear()
        self._full_history.clear()
        self.extracted_facts.clear()
        self.tool_results.clear()
        self.summary = ""


@dataclass
class AgentStep:
    """A single step in the agent's reasoning loop."""
    step_type: str  # "think", "act", "observe", "respond"
    content: str
    tool_name: Optional[str] = None
    tool_result: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    """Complete result from agent processing."""
    response: str
    intent: AgentIntent
    steps: List[AgentStep] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    tool_responses: List[ToolResponse] = field(default_factory=list)
    safety_check: Optional[SafetyCheckResult] = None
    quality_score: float = 0.0
    confidence: str = "medium"
    citations: List[str] = field(default_factory=list)
    disclaimer: str = ""
    processing_time_seconds: float = 0.0


class IntentRouter:
    """Route queries to the appropriate agent/handler based on intent."""

    INTENT_KEYWORDS = {
        AgentIntent.CRISIS: ["suicide", "self-harm", "kill myself", "end my life", "overdose"],
        AgentIntent.GENERAL: [],  # catch-all for ambiguous queries handled below
        AgentIntent.LAB_INTERPRETATION: ["lab", "test result", "blood work", "CRP", "ESR",
                                          "anti-CCP", "ANA", "RF value", "calprotectin"],
        AgentIntent.DRUG_INFO: ["drug", "medication", "methotrexate", "adalimumab",
                                 "infliximab", "side effect", "interaction", "DMARD"],
        AgentIntent.DIAGNOSIS: ["diagnos", "criteria", "differential", "present with",
                                 "symptoms suggest", "classify", "workup"],
        AgentIntent.TREATMENT: ["treat", "therapy", "management", "biologic",
                                 "remission", "protocol", "first-line"],
        AgentIntent.GUIDELINES: ["guideline", "recommendation", "standard of care",
                                  "ACR", "EULAR", "AGA", "ECCO"],
    }

    def classify(self, query: str) -> AgentIntent:
        query_lower = query.lower()

        # Ambiguous / too-vague queries → GENERAL (triggers clarification request)
        if len(query.split()) < 4:
            return AgentIntent.GENERAL

        for intent, keywords in self.INTENT_KEYWORDS.items():
            if intent == AgentIntent.GENERAL:
                continue
            if any(kw in query_lower for kw in keywords):
                return intent

        return AgentIntent.MEDICAL_QA


class MedicalAgent:
    """
    Core medical AI agent with reliability controls.

    Implements a simplified ReAct loop:
    1. Classify intent
    2. Check safety
    3. Retrieve context (RAG)
    4. Use tools if needed
    5. Generate response with prompt template
    6. Evaluate and validate output
    7. Apply safety filters
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        rag_pipeline: RAGPipeline,
        safety_pipeline: SafetyPipeline,
        monitoring: MonitoringService,
        name: str = "Dr. Immunity",
    ):
        self.name = name
        self.tools = tool_registry
        self.rag = rag_pipeline
        self.safety = safety_pipeline
        self.monitoring = monitoring
        self.router = IntentRouter()
        self.quality_checker = ResponseQualityChecker()
        self.groundedness_checker = GroundednessChecker()
        self.memory = ConversationMemory()
        self._generate_fn = None

    def set_generator(self, generate_fn):
        """Set the LLM generation function. Signature: (prompt: str) -> str"""
        self._generate_fn = generate_fn

    def process(self, query: str, context: Optional[Dict] = None, history: Optional[List] = None) -> AgentResult:
        """Process a query through the full reliability pipeline."""
        start_time = time.time()
        steps = []
        tool_responses = []

        trace = self.monitoring.create_trace(query, "immune")

        # Step 1: Classify intent
        intent = self.router.classify(query)
        trace.query_type = intent.value
        steps.append(AgentStep(step_type="think", content=f"Classified intent: {intent.value}"))

        # Step 2: Safety check on input
        input_safety = self.safety.check_input(query)
        trace.input_safety_check = input_safety.to_dict()

        if input_safety.level.value == "blocked":
            crisis_response = self.safety.get_crisis_response()
            return AgentResult(
                response=crisis_response,
                intent=intent,
                steps=steps,
                safety_check=input_safety,
                confidence="high",
                processing_time_seconds=time.time() - start_time,
            )

        sanitized_query = self.safety.sanitize_input(query)
        steps.append(AgentStep(step_type="think", content="Input safety check passed"))

        # Steps 3-4: ReAct loop — tool use + retrieval, up to MAX_REACT_ITERS
        MAX_REACT_ITERS = 3
        tool_results_text = ""
        retrieval = None

        for react_iter in range(MAX_REACT_ITERS):
            # --- Tool selection (intent-driven) ---
            tool_result = None
            if intent == AgentIntent.LAB_INTERPRETATION and react_iter == 0:
                tool_result = self._use_lab_tool(sanitized_query)
                tool_label = "lookup_lab_reference"
            elif intent == AgentIntent.DRUG_INFO and react_iter == 0:
                tool_result = self._use_drug_tool(sanitized_query)
                tool_label = "check_drug_info"
            elif intent == AgentIntent.GUIDELINES and react_iter == 0:
                tool_result = self._use_guidelines_tool(sanitized_query)
                tool_label = "search_clinical_guidelines"
            else:
                tool_label = None

            if tool_result:
                tool_responses.append(tool_result)
                tool_results_text += "\n" + tool_result.to_context_string()
                steps.append(AgentStep(step_type="act", content=f"Used {tool_label}",
                                       tool_name=tool_label, tool_result=tool_result.to_dict()))
                steps.append(AgentStep(step_type="observe", content=f"Tool returned: {tool_result.status.value}"))

            # --- RAG retrieval ---
            retrieval_query = sanitized_query
            retrieval = self.rag.retrieve(retrieval_query)
            steps.append(AgentStep(step_type="observe",
                                   content=f"[iter {react_iter+1}] Retrieved {len(retrieval.chunks)} chunks "
                                           f"(quality: {retrieval.quality_score:.2f})"))

            # --- Decide whether to continue ---
            # Stop if context is sufficient or last iteration
            if retrieval.sufficient_context or react_iter == MAX_REACT_ITERS - 1:
                break
            # Widen query on next iteration using extracted facts
            facts = self.memory.extracted_facts
            if facts.get("condition"):
                sanitized_query = f"{sanitized_query} {facts['condition']}"
            steps.append(AgentStep(step_type="think",
                                   content=f"Context insufficient (quality={retrieval.quality_score:.2f}), refining query"))

        # Ensure retrieval is always set
        if retrieval is None:
            retrieval = self.rag.retrieve(sanitized_query)
        trace.retrieval_result = {
            "chunks_found": len(retrieval.chunks),
            "quality_score": retrieval.quality_score,
            "sufficient": retrieval.sufficient_context,
            "method": retrieval.retrieval_method,
        }
        steps.append(AgentStep(step_type="observe",
                               content=f"Retrieved {len(retrieval.chunks)} chunks (quality: {retrieval.quality_score:.2f})"))

        # Step 5: Build prompt and generate response
        prompt_template = select_prompt(classify_query_type(sanitized_query))
        trace.prompt_template_used = type(prompt_template).__name__

        full_context = retrieval.context_text
        if tool_results_text:
            full_context = f"## Tool Results\n{tool_results_text}\n\n{full_context}"

        conversation_history = self.memory.get_context_summary()
        rendered_prompt = prompt_template.render(
            user_query=sanitized_query,
            context=full_context,
            history=conversation_history,
        )

        if self._generate_fn:
            response = self._generate_fn(rendered_prompt)
        else:
            response = self._fallback_response(intent, sanitized_query, retrieval)

        trace.response = response
        steps.append(AgentStep(step_type="respond", content=f"Generated response ({len(response)} chars)"))

        # Step 6: Evaluate response quality
        quality_result = self.quality_checker.evaluate(response, sanitized_query)
        groundedness_score, grounding_issues = self.groundedness_checker.check(response, full_context)

        trace.groundedness_score = groundedness_score
        trace.overall_quality_score = quality_result.overall_score
        trace.evaluation_result = {
            "quality_score": quality_result.overall_score,
            "groundedness_score": groundedness_score,
            "has_citation": quality_result.has_citation,
            "has_disclaimer": quality_result.has_disclaimer,
            "issues": quality_result.issues + grounding_issues,
        }

        # Step 7: Safety check on output
        output_safety = self.safety.check_output(response, full_context)
        trace.output_safety_check = output_safety.to_dict()

        # Step 8: Sanitize output
        final_response = self.safety.sanitize_output(response)

        # Update memory
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", final_response[:500])

        # Determine confidence
        confidence = self._assess_confidence(retrieval, quality_result, groundedness_score)

        processing_time = time.time() - start_time
        trace.generation_time_seconds = processing_time

        # Complete monitoring trace
        self.monitoring.complete_trace(trace)

        return AgentResult(
            response=final_response,
            intent=intent,
            steps=steps,
            retrieval=retrieval,
            tool_responses=tool_responses,
            safety_check=output_safety,
            quality_score=quality_result.overall_score,
            confidence=confidence,
            citations=retrieval.citations,
            disclaimer=self.safety.config.medical_disclaimer if self.safety.config.enable_medical_disclaimer else "",
            processing_time_seconds=processing_time,
        )

    def _use_lab_tool(self, query: str) -> Optional[ToolResponse]:
        import re
        lab_tests = ["CRP", "ESR", "RF", "anti-CCP", "ANA", "anti-dsDNA",
                     "calprotectin", "complement", "C3", "C4"]
        for test in lab_tests:
            if test.lower() in query.lower():
                value_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg|IU|U|ug|mm)', query)
                value = value_match.group(1) if value_match else None
                return self.tools.execute("lookup_lab_reference", lab_name=test, value=value)
        return None

    def _use_drug_tool(self, query: str) -> Optional[ToolResponse]:
        from pipeline.lib.tools import DRUG_INTERACTIONS
        query_lower = query.lower()
        for drug in DRUG_INTERACTIONS:
            if drug.lower() in query_lower:
                return self.tools.execute("check_drug_info", drug_name=drug)
        return None

    def _use_guidelines_tool(self, query: str) -> Optional[ToolResponse]:
        conditions = {
            "rheumatoid arthritis": "rheumatoid_arthritis",
            "crohn": "crohns_disease",
            "lupus": "lupus",
        }
        for condition, key in conditions.items():
            if condition.lower() in query.lower():
                return self.tools.execute("search_clinical_guidelines", condition=key)
        return None

    def _fallback_response(self, intent: AgentIntent, query: str, retrieval: RetrievalResult) -> str:
        """Generate a structured fallback response when no LLM is available."""
        # Detect ambiguous / too-short queries and ask for clarification
        if len(query.split()) < 5 or query.strip().rstrip("?.!") in (
            "is it bad", "what is it", "how is it", "tell me more", "explain", "why"
        ):
            return (
                "Could you please provide more specific information? "
                "What condition, symptom, or treatment are you asking about? "
                "For example: 'Is rheumatoid arthritis serious?' or "
                "'What does a high CRP level mean for my condition?'\n\n"
                "The more detail you provide, the more specific and useful my response can be."
            )

        parts = [f"**Summary**: Based on available medical literature regarding your question about {intent.value}.\n"]

        if retrieval.chunks:
            parts.append("**Relevant Information**:")
            for i, chunk in enumerate(retrieval.chunks[:3], 1):
                snippet = chunk.text[:300].strip()
                parts.append(f"\n{i}. From *{chunk.paper_title}* ({chunk.section}):\n   {snippet}...")

        if retrieval.citations:
            parts.append("\n**Sources**:")
            for citation in retrieval.citations[:3]:
                parts.append(f"- {citation}")

        if not retrieval.sufficient_context:
            parts.append(
                "\n**Confidence**: LOW - Insufficient context found in the knowledge base. "
                "Please consult your healthcare provider for detailed guidance."
            )
        else:
            parts.append("\n**Confidence**: MEDIUM - Based on retrieved medical literature.")

        parts.append(
            "\n**Next Steps**: Please discuss these findings with your healthcare provider "
            "for personalized medical advice."
        )

        return "\n".join(parts)

    def _assess_confidence(self, retrieval: RetrievalResult, quality, groundedness: float) -> str:
        if not retrieval.sufficient_context:
            return "low"
        if groundedness >= 0.8 and quality.overall_score >= 0.7:
            return "high"
        if groundedness >= 0.5 and quality.overall_score >= 0.4:
            return "medium"
        return "low"


class MultiAgentOrchestrator:
    """
    Multi-agent orchestration with shared state and routing.

    Coordinates specialist agents based on query intent,
    manages shared state across agents, and handles fallbacks.
    """

    def __init__(self, monitoring: MonitoringService):
        self.agents: Dict[str, MedicalAgent] = {}
        self.monitoring = monitoring
        self.router = IntentRouter()
        self.shared_state: Dict[str, Any] = {}

    def register_agent(self, name: str, agent: MedicalAgent):
        self.agents[name] = agent

    def route_and_process(self, query: str, doctor_type: str = "immune",
                          context: Optional[Dict] = None) -> AgentResult:
        """Route query to appropriate agent and process."""
        intent = self.router.classify(query)

        agent_name = self._select_agent(intent, doctor_type)
        agent = self.agents.get(agent_name)

        if agent is None:
            agent = next(iter(self.agents.values())) if self.agents else None

        if agent is None:
            return AgentResult(
                response="No specialist agent is currently available. Please try again later.",
                intent=intent,
                confidence="low",
            )

        result = agent.process(query, context)

        self.shared_state["last_intent"] = intent.value
        self.shared_state["last_agent"] = agent_name
        self.shared_state["last_confidence"] = result.confidence

        return result

    def _select_agent(self, intent: AgentIntent, doctor_type: str) -> str:
        if doctor_type in self.agents:
            return doctor_type

        intent_to_agent = {
            AgentIntent.MEDICAL_QA: "immune",
            AgentIntent.DIAGNOSIS: "immune",
            AgentIntent.TREATMENT: "immune",
            AgentIntent.LAB_INTERPRETATION: "immune",
            AgentIntent.DRUG_INFO: "immune",
            AgentIntent.GUIDELINES: "immune",
            AgentIntent.GENERAL: "immune",
            AgentIntent.CRISIS: "immune",
        }

        return intent_to_agent.get(intent, "immune")
