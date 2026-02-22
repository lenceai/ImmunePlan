"""
Chapter 8: Multi-Agent System Concepts

Demonstrates multi-agent orchestration patterns:
  - Intent-aware routing
  - Specialist agents
  - Shared state management
  - Workflow-level testing
  - Coordination reliability

Key insight: "Multi-agent reliability depends on coordination and state
management, not just individual agent quality."
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AgentSpecialty(Enum):
    MEDICAL_QA = "medical_qa"
    LAB_ANALYSIS = "lab_analysis"
    TREATMENT_INFO = "treatment_info"
    IMAGING_ANALYSIS = "imaging_analysis"


@dataclass
class SharedState:
    """
    Shared state across agents in a multi-agent workflow.

    Critical for coordination reliability:
    - All agents read/write to same state
    - State schema is defined upfront
    - State transitions are logged
    """
    patient_context: Dict = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    routing_decisions: List[Dict] = field(default_factory=list)
    current_agent: Optional[str] = None

    def add_routing_decision(self, query: str, selected_agent: str, reasoning: str):
        self.routing_decisions.append({
            "query": query,
            "selected_agent": selected_agent,
            "reasoning": reasoning,
        })
        self.current_agent = selected_agent


class IntentRouter:
    """Route queries to specialist agents based on intent."""

    ROUTING_RULES = {
        AgentSpecialty.LAB_ANALYSIS: [
            "lab", "test result", "blood work", "CRP", "ESR", "ANA",
            "anti-CCP", "RF", "calprotectin"
        ],
        AgentSpecialty.TREATMENT_INFO: [
            "treatment", "therapy", "medication", "drug", "DMARD",
            "biologic", "remission", "dose"
        ],
        AgentSpecialty.IMAGING_ANALYSIS: [
            "x-ray", "MRI", "ultrasound", "imaging", "erosion",
            "CT scan", "radiograph"
        ],
    }

    def route(self, query: str) -> AgentSpecialty:
        query_lower = query.lower()
        for specialty, keywords in self.ROUTING_RULES.items():
            if any(kw in query_lower for kw in keywords):
                return specialty
        return AgentSpecialty.MEDICAL_QA


def demonstrate_multi_agent_workflow():
    """Show a multi-agent workflow with shared state."""
    state = SharedState()
    router = IntentRouter()

    queries = [
        "My CRP is 45 mg/L and anti-CCP is positive at 120 U/mL",
        "What treatment should I consider for RA?",
        "Does my X-ray show any erosions?",
        "What does my overall picture suggest?",
    ]

    print("Multi-Agent Workflow Simulation:")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        specialty = router.route(query)
        reasoning = f"Keywords matched {specialty.value} routing rules"

        state.add_routing_decision(query, specialty.value, reasoning)

        print(f"\n  Step {i}: \"{query}\"")
        print(f"    Routed to: {specialty.value}")
        print(f"    Reasoning: {reasoning}")
        print(f"    Shared state updated: {len(state.routing_decisions)} decisions")


def workflow_testing_checklist():
    """
    Workflow-level testing checklist from Chapter 8.

    Testing must include workflow-level reliability:
    - Blended queries / multi-intent interactions
    - Ambiguous intent and routing decisions
    - Agent failure and resilience
    - State flow / information preservation
    - Regression testing
    """
    tests = {
        "Blended Query": "User asks about both labs AND treatment in one message",
        "Ambiguous Intent": "Query could be medical_qa OR treatment (e.g., 'what about methotrexate?')",
        "Agent Failure": "What happens when the specialist agent fails or times out?",
        "State Preservation": "Does context from agent A flow correctly to agent B?",
        "Regression": "Does fixing routing for one query type break another?",
    }

    print("\nWorkflow Testing Checklist:")
    for test_name, description in tests.items():
        print(f"  [ ] {test_name}: {description}")


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 8: Multi-Agent System Concepts")
    print("=" * 80)
    print()

    demonstrate_multi_agent_workflow()
    workflow_testing_checklist()

    print("\n\nKey Takeaways:")
    print("  1. Move to multi-agent only when specialization reduces complexity")
    print("  2. Focus reliability work on routing and shared state")
    print("  3. Test coordination, not just individual agents")
    print("  4. Define fallback behavior for routing failures")
