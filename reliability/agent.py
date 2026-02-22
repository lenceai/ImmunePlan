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

from reliability.tools import ToolRegistry, ToolResponse
from reliability.rag import RAGPipeline, RetrievalResult
from reliability.prompts import select_prompt, classify_query_type, PromptTemplate
from reliability.safety import SafetyPipeline, SafetyCheckResult
from reliability.evaluation import ResponseQualityChecker, GroundednessChecker
from reliability.monitoring import MonitoringService, RequestTrace

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
    Session memory for maintaining conversation context.

    Manages:
    - Short-term context (current session)
    - Key facts extracted from conversation
    - Tool results history
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    extracted_facts: Dict[str, str] = field(default_factory=dict)
    tool_results: List[Dict] = field(default_factory=list)
    session_id: str = ""
    max_history: int = 10

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history:]

    def add_tool_result(self, tool_name: str, result: Dict):
        self.tool_results.append({
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    def extract_fact(self, key: str, value: str):
        self.extracted_facts[key] = value

    def get_context_summary(self) -> str:
        parts = []
        if self.messages:
            recent = self.messages[-4:]
            for msg in recent:
                parts.append(f"{msg['role']}: {msg['content'][:200]}")

        if self.extracted_facts:
            facts = ", ".join(f"{k}: {v}" for k, v in self.extracted_facts.items())
            parts.append(f"Known facts: {facts}")

        return "\n".join(parts) if parts else ""

    def clear(self):
        self.messages.clear()
        self.extracted_facts.clear()
        self.tool_results.clear()


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

        for intent, keywords in self.INTENT_KEYWORDS.items():
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

        # Step 3: Use tools based on intent
        tool_results_text = ""
        if intent == AgentIntent.LAB_INTERPRETATION:
            tool_result = self._use_lab_tool(sanitized_query)
            if tool_result:
                tool_responses.append(tool_result)
                tool_results_text = tool_result.to_context_string()
                steps.append(AgentStep(step_type="act", content="Used lab lookup tool",
                                       tool_name="lookup_lab_reference", tool_result=tool_result.to_dict()))

        elif intent == AgentIntent.DRUG_INFO:
            tool_result = self._use_drug_tool(sanitized_query)
            if tool_result:
                tool_responses.append(tool_result)
                tool_results_text = tool_result.to_context_string()
                steps.append(AgentStep(step_type="act", content="Used drug info tool",
                                       tool_name="check_drug_info", tool_result=tool_result.to_dict()))

        elif intent == AgentIntent.GUIDELINES:
            tool_result = self._use_guidelines_tool(sanitized_query)
            if tool_result:
                tool_responses.append(tool_result)
                tool_results_text = tool_result.to_context_string()
                steps.append(AgentStep(step_type="act", content="Used guidelines tool",
                                       tool_name="search_clinical_guidelines", tool_result=tool_result.to_dict()))

        # Step 4: RAG retrieval
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
        drugs = ["methotrexate", "adalimumab", "infliximab"]
        for drug in drugs:
            if drug.lower() in query.lower():
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
