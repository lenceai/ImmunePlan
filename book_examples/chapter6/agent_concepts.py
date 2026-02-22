"""
Chapter 6: Agent Concepts - Creating Effective AI Agents

Demonstrates the key concepts from Chapter 6:
  - Agent vs Chatbot distinction
  - Memory (short-term and persistent)
  - Tool integration with validation
  - ReAct reasoning loop (reason -> act -> observe)
  - Anti-hallucination guardrails for agents

Key insight: "Agents multiply capability and failure modes together."
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    """
    Agent memory management.

    Short-term: current interaction context
    Persistent: long-term preferences/history

    Reliability implications:
    - Stale memory can mislead
    - Contradictory memory must be handled
    - Privacy concerns with persistent storage
    """
    short_term: List[Dict] = field(default_factory=list)
    persistent: Dict[str, str] = field(default_factory=dict)

    def add_turn(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-10:]

    def remember(self, key: str, value: str):
        self.persistent[key] = value

    def recall(self, key: str) -> Optional[str]:
        return self.persistent.get(key)


@dataclass
class ReActStep:
    """Single step in the ReAct reasoning loop."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


def react_loop_example():
    """
    Demonstrate the ReAct reasoning loop.

    The loop:
    1. Thought: reason about what to do
    2. Action: choose and execute a tool
    3. Observation: process the tool result
    4. Repeat or respond
    """
    query = "What is the current CRP level interpretation for a value of 45 mg/L?"

    steps = []

    step1 = ReActStep(
        thought="The user is asking about CRP interpretation. I need to look up the reference range and interpret the value.",
        action="lookup_lab_reference",
        action_input="CRP, value=45",
        observation="CRP reference: normal <3 mg/L, elevated 3-10, high >10. Value 45 is significantly elevated."
    )
    steps.append(step1)

    step2 = ReActStep(
        thought="CRP of 45 is significantly elevated, indicating active inflammation. I should provide context about what this means for autoimmune conditions.",
    )
    steps.append(step2)

    print("ReAct Loop Execution:")
    for i, step in enumerate(steps, 1):
        print(f"\n  Step {i}:")
        print(f"    Thought: {step.thought}")
        if step.action:
            print(f"    Action: {step.action}({step.action_input})")
        if step.observation:
            print(f"    Observation: {step.observation}")


def agent_guardrails_example():
    """
    Demonstrate anti-hallucination guardrails for agents.

    Key guardrails:
    1. Cross-referencing tool results with known data
    2. Validating tool inputs before execution
    3. Constraining action permissions
    4. Step-level observability
    """
    print("\nAgent Guardrails:")

    guardrails = {
        "Input Validation": "Validate tool inputs against expected schemas before execution",
        "Action Permissions": "Only execute tools from the allowed list; never modify patient records",
        "Output Verification": "Cross-reference generated claims against retrieved sources",
        "Uncertainty Handling": "When tool returns empty/ambiguous results, state uncertainty explicitly",
        "Step Logging": "Log every thought, action, and observation for auditability",
    }

    for name, description in guardrails.items():
        print(f"  [{name}]: {description}")


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 6: Agent Concepts")
    print("=" * 80)
    print()

    print("--- Agent vs Chatbot ---")
    print("  Chatbot: answers questions")
    print("  Agent: retrieves, decides, acts, follows up")
    print("  Warning: avoid 'agent theater' - use agents only when truly needed")

    print("\n--- ReAct Loop ---")
    react_loop_example()

    print("\n--- Guardrails ---")
    agent_guardrails_example()

    print("\n--- Memory Policy ---")
    memory = AgentMemory()
    memory.add_turn("user", "I have RA and I'm on methotrexate")
    memory.remember("condition", "Rheumatoid Arthritis")
    memory.remember("current_medication", "Methotrexate")
    print(f"  Stored: condition={memory.recall('condition')}, medication={memory.recall('current_medication')}")
    print("  Policy: Clear PII after session; retain only condition/medication context")
