"""
OMAC: Optimization Framework for LLM-Based Multi-Agent Collaboration
=====================================================================

This code provides a working illustration of the key concepts from the OMAC paper
(Li, Hasson, Ghosh — UT Austin & Intuit AI Research, 2025).

OMAC addresses a fundamental question: instead of hand-crafting multi-agent systems,
can we *systematically optimize* both what agents do and how they collaborate?

The framework identifies 5 optimization dimensions:
  - Fun-1: Optimize existing agent prompts
  - Fun-2: Construct new agents for the team
  - Str-1: Select which agents form the candidate team
  - Str-2: Dynamically pick agents per collaboration step
  - Str-3: Optimize communication patterns between agents

It uses two key actors:
  - Semantic Initializer: generates diverse initial agent/controller configurations
  - Contrastive Comparator: compares high/low performers to generate improvements

NOTE: This is an educational implementation. In practice, you'd use actual LLM API
calls. Here we simulate the LLM interactions to make the concepts concrete.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod


# =============================================================================
# PART 1: Core Abstractions — Agents and the Collaboration Process
# =============================================================================

@dataclass
class Agent:
    """
    An LLM-powered agent, as defined in the paper (Definition 1).
    
    An agent is a function A: P × I → O, where:
      - P = instruction prompt (defines role + behavior)
      - I = input (query, task description, outputs from other agents)
      - O = output (analyses, suggestions, solutions)
    
    In OMAC, the prompt is the primary optimization target. By refining what
    instructions an agent receives, we change how it approaches problems.
    """
    name: str
    role: str
    instruction_prompt: str
    few_shot_examples: List[Dict] = field(default_factory=list)
    
    # In a real system, this would call an LLM API
    def generate(self, task_input: str, context: List[str] = None) -> str:
        """
        Simulate the agent generating a response.
        
        In practice, this would be:
            response = llm.complete(
                system=self.instruction_prompt,
                messages=[{"role": "user", "content": task_input}]
            )
        """
        context_str = f" (with {len(context)} prior solutions)" if context else ""
        return f"[{self.name}/{self.role}]{context_str}: Solution for '{task_input[:50]}...'"
    
    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"


@dataclass
class CollaborationStep:
    """
    One step in the multi-step collaboration process (Definition 2).
    
    Each step involves a subset of agents from the overall team. Agents at
    step t can leverage solutions from steps 1..t-1. This is the key insight
    that makes multi-agent collaboration powerful — later agents build on
    earlier work.
    
    Example in code generation:
      Step 1: Programmer writes initial code
      Step 2: Tester generates unit tests
      Step 3: Reviewer analyzes test results + code
      Step 4: Programmer rewrites based on feedback
    """
    step_number: int
    participating_agents: List[Agent]
    outputs: Dict[str, str] = field(default_factory=dict)


class MultiAgentSystem:
    """
    The core MAS that OMAC optimizes.
    
    This implements the multi-step collaboration process where agents
    work together sequentially. The collaboration structure — who participates,
    when, and who receives whose output — is exactly what OMAC's structural
    dimensions (Str-1, Str-2, Str-3) optimize.
    """
    
    def __init__(self, agents: List[Agent], max_steps: int = 4):
        self.agents = agents
        self.max_steps = max_steps
        # Controllers manage the structural aspects
        self.candidate_selector = None    # Str-1: picks the team
        self.step_selector = None         # Str-2: picks agents per step
        self.communication_router = None  # Str-3: routes messages between agents
    
    def run_collaboration(self, task: str) -> List[CollaborationStep]:
        """
        Execute the multi-step collaboration on a task.
        
        This is the process that gets evaluated during OMAC optimization.
        The quality of the final output depends on both the agents (functional)
        and how they're orchestrated (structural).
        """
        # Str-1: Select candidate agents for this task
        candidates = self._select_candidates(task)
        
        steps = []
        all_previous_outputs = []
        
        for step_num in range(1, self.max_steps + 1):
            # Str-2: Dynamically select agents for this step
            step_agents = self._select_step_agents(
                candidates, step_num, all_previous_outputs, task
            )
            
            step = CollaborationStep(step_number=step_num, participating_agents=step_agents)
            
            for agent in step_agents:
                # Str-3: Determine which previous outputs this agent should see
                relevant_context = self._route_communication(
                    agent, all_previous_outputs, task
                )
                
                # Agent generates its solution
                output = agent.generate(task, context=relevant_context)
                step.outputs[agent.name] = output
                all_previous_outputs.append((agent.name, output))
            
            steps.append(step)
        
        return steps
    
    def _select_candidates(self, task: str) -> List[Agent]:
        """Str-1: Select candidate agents from the full pool."""
        if self.candidate_selector:
            return self.candidate_selector.select(self.agents, task)
        return self.agents  # Default: use all agents
    
    def _select_step_agents(
        self, candidates: List[Agent], step: int,
        previous_outputs: List, task: str
    ) -> List[Agent]:
        """Str-2: Dynamically select agents for this collaboration step."""
        if self.step_selector:
            return self.step_selector.select(candidates, step, previous_outputs, task)
        return candidates  # Default: all candidates participate every step
    
    def _route_communication(
        self, agent: Agent, previous_outputs: List, task: str
    ) -> List[str]:
        """Str-3: Decide which previous outputs to route to this agent."""
        if self.communication_router:
            return self.communication_router.route(agent, previous_outputs, task)
        # Default: fully connected — agent sees everything
        return [output for _, output in previous_outputs]


# =============================================================================
# PART 2: The Two Key Actors — Semantic Initializer & Contrastive Comparator
# =============================================================================

class SemanticInitializer:
    """
    The Semantic Initializer (Section 3.2).
    
    This LLM-powered actor generates an initial DIVERSE collection of
    agent prompts or controller configurations. The key idea is to explore
    the semantic space — generating variations that differ in focus,
    detail level, and approach.
    
    For example, when generating "Programmer" agent prompts, it might produce:
      1. A minimal prompt: "Write Python code for the given function."
      2. A detailed prompt: "Write clean, well-commented Python code..."
      3. A structured prompt: "First analyze the docstring, then plan your
         approach, then implement with error handling..."
    
    Each variation explores a different hypothesis about what makes a good prompt.
    """
    
    def __init__(self, collection_size: int = 3):
        self.collection_size = collection_size
    
    def generate_agent_variants(
        self, role: str, task_context: str, base_prompt: str
    ) -> List[str]:
        """
        Generate diverse prompt variants for an agent role.
        
        In practice, this calls an LLM with a meta-prompt like:
        
            "Generate {n} distinct prompts to instruct an LLM to resolve
             [task] acting as [role]. Each prompt should guide the model
             to accurately resolve problems while adhering to the role.
             Ensure the prompts differ in content and structure..."
        
        The paper provides the full prompt templates in Appendix C.1.
        """
        # Simulated diverse prompts — in reality, an LLM would generate these
        variants = []
        
        # Variant 1: Minimal/direct approach
        variants.append(
            f"You are a {role}. {base_prompt} "
            f"Focus on correctness and provide your solution directly."
        )
        
        # Variant 2: Structured/methodical approach
        variants.append(
            f"You are an experienced {role}. {base_prompt} "
            f"First analyze the problem carefully, then plan your approach "
            f"step by step, and finally implement your solution with clear "
            f"comments explaining each decision."
        )
        
        # Variant 3: Quality-focused approach
        variants.append(
            f"You are an expert {role}. {base_prompt} "
            f"Prioritize code readability, include comprehensive comments, "
            f"handle edge cases explicitly, and verify your solution against "
            f"the requirements before finalizing."
        )
        
        return variants[:self.collection_size]
    
    def generate_controller_variants(
        self, controller_type: str, task_context: str
    ) -> List[str]:
        """
        Generate diverse controller prompt variants for structural optimization.
        
        Controllers manage the collaboration structure. For example, a Str-1
        controller decides which agents should be on the team.
        """
        variants = []
        
        if controller_type == "candidate_selection":  # Str-1
            variants = [
                "Select the top 5 agents best suited for this task based on "
                "their functional relevance to the problem domain.",
                
                "Choose agents whose expertise areas most directly relate to "
                "the task. Prioritize diversity of perspectives. Select 4 agents.",
                
                "Evaluate each agent's potential contribution. Select 6 agents, "
                "emphasizing both core expertise and complementary skills."
            ]
        
        elif controller_type == "dynamic_participation":  # Str-2
            variants = [
                "Select the 3 agents whose expertise is most relevant to the "
                "current step, considering what has already been produced.",
                
                "Choose agents that can best build on previous solutions. "
                "In early steps, favor broad thinkers; in later steps, favor specialists.",
                
                "Analyze previous outputs and select agents who can address "
                "gaps or weaknesses in the current solution set."
            ]
        
        elif controller_type == "communication":  # Str-3
            variants = [
                "Route each agent's output only to agents whose roles can "
                "directly benefit from that information.",
                
                "For code tasks: route Programmer output to Tester and Reviewer; "
                "route Tester output back to Programmer only.",
                
                "Select the top 4 most relevant source agents for each target "
                "agent, based on functional complementarity."
            ]
        
        return variants[:self.collection_size]


class ContrastiveComparator:
    """
    The Contrastive Comparator (Section 3.2).
    
    This is the core optimization engine of OMAC. Given a positive-negative
    pair (a high-performing and low-performing prompt/controller), it:
    
      1. Analyzes WHY the positive outperforms the negative
      2. Identifies key differentiating factors
      3. Generates a NEW prompt that amplifies positive factors
         and removes negative factors
    
    This is fundamentally a form of "verbal reinforcement learning" — the
    performance gap provides a supervised signal, and the LLM reasons about
    it to produce improvements.
    
    Example from the paper (Figure 2):
      Positive prompt: "Include comments throughout your code..."  → score 0.913
      Negative prompt: "You must complete the python function..."  → score 0.895
      
      Comparator reasoning: "Comments appear to be a key factor."
      Generated prompt: "Include comments that clarify functionality of each
                         line, making code more understandable and logical..."
    """
    
    def compare_and_refine(
        self,
        positive_prompt: str,
        negative_prompt: str,
        positive_score: float,
        negative_score: float,
        task_context: str,
        role: str = None
    ) -> str:
        """
        Perform contrastive reasoning and generate a refined prompt.
        
        In practice, this calls an LLM with a meta-prompt like:
        
            "A pair of parent prompts is provided: one positive and one negative.
             The positive parent prompt has been shown to be more effective.
             
             Your task is to carefully compare the two parent prompts,
             identifying the key reasons why the positive performs better.
             Based on these insights, generate a child prompt that further
             improves upon the positive parent prompt."
        
        The paper provides full prompt templates in Appendix C.1, Table 11.
        """
        # Simulated contrastive reasoning — in reality an LLM does this analysis
        
        # Step 1: Identify differentiating factors
        pos_features = set(positive_prompt.lower().split()) 
        neg_features = set(negative_prompt.lower().split())
        unique_to_positive = pos_features - neg_features
        
        # Step 2: Generate a refined prompt that amplifies positive factors
        # In practice, the LLM would do sophisticated reasoning here
        refined = (
            f"{positive_prompt} "
            f"Additionally, ensure thorough analysis before responding. "
            f"Double-check your work for accuracy and completeness. "
            f"Explain your reasoning process clearly."
        )
        
        return refined


# =============================================================================
# PART 3: The OMAC Optimization Algorithm
# =============================================================================

class OMACOptimizer:
    """
    The main OMAC optimization framework.
    
    Implements both single-dimension optimization (Section 3.2) and
    multi-dimension joint optimization (Section 3.3).
    
    The optimization loop for a single dimension:
      1. Semantic Initializer generates diverse initial collection
      2. Each variant is evaluated through the full MAS collaboration
      3. Performance scores are computed
      4. A positive-negative pair is sampled
      5. Contrastive Comparator generates a refined variant
      6. Steps 2-5 repeat for max_iterations
      7. Best-performing variant is selected
    """
    
    def __init__(
        self,
        mas: MultiAgentSystem,
        evaluator: Callable,          # Function to score MAS output
        collection_size: int = 3,     # Size of initial collection (z)
        max_iterations: int = 3,      # Max contrastive iterations (w)
        threshold_h: float = 0.5,     # Upper threshold for positive sampling
        threshold_l: float = 0.5,     # Lower threshold for negative sampling
    ):
        self.mas = mas
        self.evaluator = evaluator
        self.collection_size = collection_size
        self.max_iterations = max_iterations
        self.threshold_h = threshold_h
        self.threshold_l = threshold_l
        self.initializer = SemanticInitializer(collection_size)
        self.comparator = ContrastiveComparator()
    
    def optimize_single_dimension(
        self,
        dimension: str,
        training_data: List[Dict],
        **kwargs
    ) -> Tuple[str, float]:
        """
        Optimize a single dimension of the MAS.
        
        This implements Algorithm 1 from the paper (Section 3.2):
        
        1. Initialize collection via Semantic Initializer
        2. Evaluate each variant on training data
        3. Sample positive-negative pair based on scores
        4. Contrastive Comparator generates refined variant
        5. Evaluate and add to collection
        6. Repeat steps 3-5 for max_iterations
        7. Return best-performing variant
        
        Args:
            dimension: One of 'Fun-1', 'Fun-2', 'Str-1', 'Str-2', 'Str-3'
            training_data: List of task examples with ground truth
            **kwargs: Dimension-specific parameters (e.g., agent_index for Fun-1)
        """
        print(f"\n{'='*60}")
        print(f"OMAC: Optimizing dimension [{dimension}]")
        print(f"{'='*60}")
        
        # ---- Step 1: Generate initial collection ----
        collection = self._initialize_collection(dimension, **kwargs)
        print(f"\n[Step 1] Semantic Initializer generated {len(collection)} variants:")
        for i, variant in enumerate(collection):
            print(f"  Variant {i+1}: '{variant[:80]}...'")
        
        # ---- Step 2: Evaluate each variant ----
        scores = self._evaluate_collection(collection, dimension, training_data, **kwargs)
        print(f"\n[Step 2] Evaluation scores:")
        for i, (variant, score) in enumerate(zip(collection, scores)):
            print(f"  Variant {i+1}: {score:.3f}")
        
        # ---- Steps 3-6: Iterative contrastive refinement ----
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[Iteration {iteration}] Contrastive Reasoning")
            
            # Step 3: Sample positive-negative pair
            pos_idx, neg_idx = self._sample_pair(scores)
            print(f"  Sampled pair: Positive=Variant {pos_idx+1} ({scores[pos_idx]:.3f}), "
                  f"Negative=Variant {neg_idx+1} ({scores[neg_idx]:.3f})")
            
            # Step 4: Contrastive Comparator generates refined variant
            refined = self.comparator.compare_and_refine(
                positive_prompt=collection[pos_idx],
                negative_prompt=collection[neg_idx],
                positive_score=scores[pos_idx],
                negative_score=scores[neg_idx],
                task_context=dimension,
                role=kwargs.get('role', None)
            )
            print(f"  Generated refined variant: '{refined[:80]}...'")
            
            # Step 5: Evaluate refined variant
            refined_score = self._evaluate_single(refined, dimension, training_data, **kwargs)
            print(f"  Refined variant score: {refined_score:.3f}")
            
            # Add to collection
            collection.append(refined)
            scores.append(refined_score)
        
        # ---- Step 7: Select best ----
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        print(f"\n[Result] Best variant (index {best_idx+1}): score={scores[best_idx]:.3f}")
        print(f"  Prompt: '{collection[best_idx][:100]}...'")
        
        return collection[best_idx], scores[best_idx]
    
    def optimize_multiple_dimensions(
        self,
        dimensions: List[str],
        training_data: List[Dict],
        joint_iterations: int = 3,
        **kwargs
    ) -> Dict[str, Tuple[str, float]]:
        """
        Joint optimization across multiple dimensions (Section 3.3).
        
        Key insight: Optimize ONE dimension at a time while keeping others
        fixed. This preserves the effectiveness of contrastive reasoning —
        when only one thing changes between positive and negative, the
        Comparator can clearly identify what caused the difference.
        
        If you changed multiple things simultaneously, the Comparator
        would face a confounding problem — was the improvement due to
        a better agent prompt or a better communication structure?
        
        The paper validates this in Appendix B.2.2 (Figure 7), showing
        that simultaneous multi-dimension optimization leads to reduced
        gains and higher variance.
        
        Algorithm:
          for each joint iteration:
              for each dimension in selected_dimensions:
                  optimize this dimension (keeping others fixed)
                  keep the best result
        """
        print(f"\n{'='*60}")
        print(f"OMAC: Joint optimization of dimensions {dimensions}")
        print(f"  Running {joint_iterations} joint iterations")
        print(f"{'='*60}")
        
        results = {}
        
        for joint_iter in range(1, joint_iterations + 1):
            print(f"\n--- Joint Iteration {joint_iter}/{joint_iterations} ---")
            
            for dim in dimensions:
                best_prompt, best_score = self.optimize_single_dimension(
                    dim, training_data, **kwargs
                )
                results[dim] = (best_prompt, best_score)
                
                # In practice: update the MAS with the optimized agent/controller
                # before moving to the next dimension
                print(f"  [{dim}] Updated MAS with optimized config (score: {best_score:.3f})")
        
        return results
    
    # ---- Helper methods ----
    
    def _initialize_collection(self, dimension: str, **kwargs) -> List[str]:
        """Route to appropriate initializer based on dimension."""
        if dimension == "Fun-1":
            return self.initializer.generate_agent_variants(
                role=kwargs.get('role', 'General'),
                task_context=kwargs.get('task_context', ''),
                base_prompt=kwargs.get('base_prompt', 'Solve the given problem.')
            )
        elif dimension == "Fun-2":
            return self.initializer.generate_agent_variants(
                role="New Specialist",
                task_context=kwargs.get('task_context', ''),
                base_prompt="Bring a unique perspective to the team."
            )
        elif dimension in ("Str-1", "Str-2", "Str-3"):
            type_map = {
                "Str-1": "candidate_selection",
                "Str-2": "dynamic_participation",
                "Str-3": "communication"
            }
            return self.initializer.generate_controller_variants(
                controller_type=type_map[dimension],
                task_context=kwargs.get('task_context', '')
            )
        return []
    
    def _evaluate_collection(
        self, collection: List[str], dimension: str,
        training_data: List[Dict], **kwargs
    ) -> List[float]:
        """Evaluate all variants in the collection."""
        return [
            self._evaluate_single(variant, dimension, training_data, **kwargs)
            for variant in collection
        ]
    
    def _evaluate_single(
        self, variant: str, dimension: str,
        training_data: List[Dict], **kwargs
    ) -> float:
        """
        Evaluate a single variant by running full MAS collaboration on training data.
        
        In practice, this:
          1. Swaps the variant into the MAS (updating the relevant agent/controller)
          2. Runs the collaboration process on each training example
          3. Computes the evaluation metric (Pass@1, accuracy, etc.)
        
        This is the most computationally expensive part of OMAC — each
        evaluation requires running the full multi-agent pipeline on the
        entire training set.
        """
        # Simulated evaluation — scores based on prompt characteristics
        # In reality, this would be actual task performance
        base_score = 0.65
        
        # Longer, more detailed prompts tend to score slightly higher
        # (This is a simplification — real scoring uses actual task metrics)
        length_bonus = min(len(variant) / 1000, 0.15)
        
        # Prompts mentioning key strategies get bonuses
        strategy_keywords = ["step by step", "analyze", "verify", "comment",
                           "edge case", "reasoning", "plan"]
        keyword_bonus = sum(0.02 for kw in strategy_keywords if kw in variant.lower())
        
        # Add controlled randomness to simulate real evaluation variance
        noise = random.gauss(0, 0.02)
        
        score = min(base_score + length_bonus + keyword_bonus + noise, 1.0)
        return max(score, 0.0)
    
    def _sample_pair(self, scores: List[float]) -> Tuple[int, int]:
        """
        Sample a positive-negative pair based on performance thresholds.
        
        The paper defines thresholds h and l (default 0.5 each):
          - Top ⌊n × h⌋ scores → positive pool
          - Bottom ⌊n × l⌋ scores → negative pool
        
        Sampling within thresholds (rather than just picking best/worst)
        diversifies the pairs, improving robustness of contrastive reasoning.
        
        The paper shows in Table 7 that OMAC is robust to threshold choices,
        with performance fluctuating within ~2%.
        """
        n = len(scores)
        sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        
        # Top performers (positive pool)
        num_positive = max(1, math.floor(n * self.threshold_h))
        positive_pool = sorted_indices[:num_positive]
        
        # Bottom performers (negative pool)
        num_negative = max(1, math.floor(n * self.threshold_l))
        negative_pool = sorted_indices[-num_negative:]
        
        # Random sample from each pool
        pos_idx = random.choice(positive_pool)
        neg_idx = random.choice(negative_pool)
        
        # Ensure they're different
        if pos_idx == neg_idx and n > 1:
            neg_idx = sorted_indices[-1] if pos_idx != sorted_indices[-1] else sorted_indices[-2]
        
        return pos_idx, neg_idx


# =============================================================================
# PART 4: Putting It All Together — A Complete Example
# =============================================================================

def run_code_generation_example():
    """
    Demonstrates OMAC optimization on a code generation task,
    mirroring the paper's HumanEval experiments.
    
    The paper uses 7 agents for code generation:
      - 4 code writers: Python Assistant, Algorithm Developer,
                        Computer Scientist, Programmer
      - 3 code reviewers: Syntax Checker, Unit Tester, Reflector
    
    Results from the paper (Table 2):
      - DyLAN baseline: 85.74% Pass@1
      - OMAC Fun-1.4 (optimizing Programmer): 89.25% Pass@1
      - OMAC Str-3 (optimizing communication): 87.55% Pass@1
    """
    print("\n" + "="*70)
    print("  OMAC Code Generation Example")
    print("  (Mirrors HumanEval experiments from the paper)")
    print("="*70)
    
    # Create the default agent team (from DyLAN configuration)
    agents = [
        Agent("agent_1", "Python Assistant",
              "You are a Python assistant. Complete the given function."),
        Agent("agent_2", "Algorithm Developer",
              "You are an algorithm developer. Implement efficient solutions."),
        Agent("agent_3", "Computer Scientist",
              "You are a computer scientist. Write correct, optimal code."),
        Agent("agent_4", "Programmer",
              "You are a programmer. Complete the python function given to you."),
        Agent("agent_5", "Syntax Checker",
              "You are a syntax checker. Review the code for errors."),
        Agent("agent_6", "Unit Tester",
              "You are a unit tester. Generate and run tests for the code."),
        Agent("agent_7", "Reflector",
              "You are a reflector. Analyze the code and suggest improvements."),
    ]
    
    # Create the multi-agent system
    mas = MultiAgentSystem(agents, max_steps=6)
    
    # Simulated training data (in practice: HumanEval problems with unit tests)
    training_data = [
        {"task": "Implement a function to check if a string is a palindrome",
         "test": "assert is_palindrome('racecar') == True"},
        {"task": "Implement a function to find the nth Fibonacci number",
         "test": "assert fibonacci(10) == 55"},
        {"task": "Implement a function to flatten a nested list",
         "test": "assert flatten([[1,[2]],3]) == [1,2,3]"},
    ]
    
    # Create the optimizer
    optimizer = OMACOptimizer(
        mas=mas,
        evaluator=lambda x: random.uniform(0.7, 0.95),  # Simplified
        collection_size=3,    # Paper default: z=3
        max_iterations=3,     # Paper default: w=3
        threshold_h=0.5,      # Paper default
        threshold_l=0.5,      # Paper default
    )
    
    # --- Example 1: Optimize the Programmer agent (Fun-1.4) ---
    print("\n\n>>> Optimizing the Programmer agent (Fun-1, agent 4)")
    print("    This is the dimension that achieved 89.25% in the paper")
    
    best_prompt, best_score = optimizer.optimize_single_dimension(
        dimension="Fun-1",
        training_data=training_data,
        role="Programmer",
        base_prompt="Complete the python function. Follow the format in the docstring.",
        task_context="code generation"
    )
    
    # --- Example 2: Optimize communication patterns (Str-3) ---
    print("\n\n>>> Optimizing communication patterns (Str-3)")
    print("    Controls which agents' outputs get routed where")
    
    best_controller, best_score = optimizer.optimize_single_dimension(
        dimension="Str-3",
        training_data=training_data,
        task_context="code generation"
    )
    
    # --- Example 3: Joint optimization of best dimensions ---
    print("\n\n>>> Joint optimization of Fun-1 + Str-3")
    print("    Paper shows this yields the biggest improvements")
    
    results = optimizer.optimize_multiple_dimensions(
        dimensions=["Fun-1", "Str-3"],
        training_data=training_data,
        joint_iterations=2,  # Paper uses 3
        role="Programmer",
        base_prompt="Complete the python function.",
        task_context="code generation"
    )


def run_ablation_example():
    """
    Demonstrates the ablation study (Section 4.3).
    
    OMAC-C removes the Contrastive Comparator, keeping only the
    Semantic Initializer. This tests whether the contrastive reasoning
    actually helps, or if diverse initialization alone is enough.
    
    Paper findings (Table 4, arithmetic reasoning):
      - DyLAN:  32.35% accuracy
      - OMAC-C: 34.20% (Initializer alone still beats DyLAN!)
      - OMAC:   35.01% (Comparator adds meaningful improvement)
    
    Key insight: Even just exploring the semantic space with diverse
    initialization helps, but contrastive reasoning adds significant value.
    """
    print("\n" + "="*70)
    print("  Ablation Study: OMAC vs OMAC-C (no Contrastive Comparator)")
    print("="*70)
    
    training_data = [{"task": f"Problem {i}", "answer": str(i)} for i in range(10)]
    
    # OMAC-C: Only Semantic Initializer (no iterative refinement)
    print("\n[OMAC-C] Using Semantic Initializer only...")
    initializer = SemanticInitializer(collection_size=3)
    variants = initializer.generate_agent_variants(
        role="Mathematician",
        task_context="arithmetic reasoning",
        base_prompt="Solve the math problem step by step."
    )
    
    # Score each variant (simplified)
    omac_c_scores = [0.65 + len(v)/1000 + random.gauss(0, 0.01) for v in variants]
    best_omac_c = max(omac_c_scores)
    print(f"  Best score from initialization alone: {best_omac_c:.3f}")
    
    # Full OMAC: Initializer + Contrastive Comparator
    print("\n[OMAC] Using full pipeline with Contrastive Comparator...")
    comparator = ContrastiveComparator()
    
    # Simulate 3 iterations of contrastive refinement
    for i in range(3):
        sorted_idx = sorted(range(len(omac_c_scores)), 
                          key=lambda x: omac_c_scores[x], reverse=True)
        pos, neg = sorted_idx[0], sorted_idx[-1]
        
        refined = comparator.compare_and_refine(
            variants[pos], variants[neg],
            omac_c_scores[pos], omac_c_scores[neg],
            "arithmetic reasoning", "Mathematician"
        )
        
        # Refined variants tend to score higher
        refined_score = omac_c_scores[pos] + random.uniform(0.005, 0.02)
        variants.append(refined)
        omac_c_scores.append(refined_score)
    
    best_omac = max(omac_c_scores)
    print(f"  Best score with contrastive refinement: {best_omac:.3f}")
    print(f"\n  Improvement from Contrastive Comparator: +{best_omac - best_omac_c:.3f}")


# =============================================================================
# PART 5: Real-World Integration Pattern
# =============================================================================

def show_real_world_pattern():
    """
    Shows how you would integrate OMAC with actual LLM APIs.
    
    This is pseudocode showing the integration points where you'd
    replace our simulated methods with real API calls.
    """
    pattern = """
    # ========================================================
    # Real-World OMAC Integration (Pseudocode)
    # ========================================================
    
    from openai import OpenAI
    client = OpenAI()
    
    # --- Real Semantic Initializer ---
    def semantic_initialize(role, task, n=3):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f'''Generate {n} distinct prompts to instruct 
                an LLM to resolve {task} problems acting as {role}.
                Each prompt should differ in focus and approach.
                Return as JSON array.'''
            }],
            temperature=0.8  # Higher temp = more diversity
        )
        return json.loads(response.choices[0].message.content)
    
    # --- Real Contrastive Comparator ---
    def contrastive_compare(positive, negative, pos_score, neg_score):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f'''Compare these two prompts:
                
                POSITIVE (score: {pos_score}): {positive}
                NEGATIVE (score: {neg_score}): {negative}
                
                Identify WHY the positive outperforms the negative.
                Generate a new prompt that improves on the positive.'''
            }]
        )
        return response.choices[0].message.content
    
    # --- Real Evaluation on Training Data ---
    def evaluate_agent_prompt(prompt, training_data, other_agents):
        total_score = 0
        for example in training_data:
            # Run full MAS collaboration with this prompt variant
            mas = build_mas_with_prompt(prompt, other_agents)
            result = mas.run_collaboration(example["task"])
            
            # Score against ground truth
            score = compute_metric(result, example["ground_truth"])
            total_score += score
        
        return total_score / len(training_data)
    
    # --- The Full OMAC Loop ---
    def omac_optimize(role, task, training_data, other_agents):
        # Step 1: Initialize
        prompts = semantic_initialize(role, task, n=3)
        scores = [evaluate_agent_prompt(p, training_data, other_agents) 
                  for p in prompts]
        
        # Steps 2-6: Iterative refinement
        for iteration in range(3):
            # Sample positive-negative pair
            pos_idx = scores.index(max(scores))
            neg_idx = scores.index(min(scores))
            
            # Contrastive reasoning
            refined = contrastive_compare(
                prompts[pos_idx], prompts[neg_idx],
                scores[pos_idx], scores[neg_idx]
            )
            
            # Evaluate and add
            refined_score = evaluate_agent_prompt(
                refined, training_data, other_agents
            )
            prompts.append(refined)
            scores.append(refined_score)
        
        # Return best
        best_idx = scores.index(max(scores))
        return prompts[best_idx], scores[best_idx]
    """
    print(pattern)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Run the code generation example
    run_code_generation_example()
    
    # Run the ablation study
    run_ablation_example()
    
    # Show real-world integration pattern
    print("\n" + "="*70)
    print("  Real-World Integration Pattern (Pseudocode)")
    print("="*70)
    show_real_world_pattern()
    
    print("\n" + "="*70)
    print("  Key Takeaways from the OMAC Paper")
    print("="*70)
    print("""
    1. STRUCTURE MATTERS: OMAC identifies 5 optimization dimensions —
       both what agents do (functional) and how they work together (structural).
    
    2. CONTRASTIVE REASONING WORKS: Comparing high vs low performers and
       reasoning about the gap is an effective optimization signal.
       The ablation study confirms the Contrastive Comparator adds
       meaningful value beyond just diverse initialization.
    
    3. OPTIMIZE ONE THING AT A TIME: Joint optimization works best when
       you iterate one dimension at a time (keeping others fixed).
       Simultaneously varying multiple dimensions confounds the
       contrastive reasoning and reduces gains.
    
    4. EVEN SIMPLE EXPLORATION HELPS: Just generating diverse prompt
       variants and picking the best (OMAC-C) already beats hand-crafted
       baselines. The full OMAC pipeline adds further improvement.
    
    5. RESULTS ARE SIGNIFICANT:
       - Code generation (HumanEval): 85.74% → 89.25% (Fun-1.4)
       - General reasoning (MMLU):    69.42% → 74.22% (Fun-1.4)
       - Arithmetic (MATH):           32.35% → 35.21% (Fun-1.1)
       - Multi-dimension optimization pushes results even higher.
    """)
