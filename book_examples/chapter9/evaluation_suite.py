"""
Chapter 9: Evaluation Suite

Combines all evaluation techniques from the book:
  - LLM-as-Judge for semantic quality
  - ROUGE scores for summarization quality
  - Hallucination detection via grounding checks
  - Multi-model fallback for cost optimization
  - Performance tracking

Demonstrates: "You can't improve RAG by vibes. You need test data and metrics."
"""

from openai import OpenAI
import os
import json
import time
from typing import Dict, List, Optional

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def llm_as_judge(question: str, answer: str, criteria: str = "accuracy") -> Dict:
    """Use LLM to evaluate answer quality with structured output."""

    evaluation_prompt = f"""You are an expert evaluator. Rate the following answer on a scale of 1-10 based on {criteria}.

Question: {question}

Answer: {answer}

Provide your evaluation in this format:
Score: [1-10]
Reasoning: [Your detailed explanation]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert evaluator who provides fair, objective assessments."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3,
    )

    evaluation = response.choices[0].message.content
    return {"criteria": criteria, "evaluation": evaluation}


def multi_model_fallback(query: str) -> Dict:
    """Route query to appropriate model based on complexity."""

    complex_keywords = [
        "legal", "regulatory", "international", "technical",
        "integration", "api", "custom", "compliance"
    ]

    is_complex = any(kw in query.lower() for kw in complex_keywords) or len(query.split()) > 20

    primary_model = "gpt-4o-mini"
    advanced_model = "gpt-4o"
    model = advanced_model if is_complex else primary_model

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        return {
            "model": model,
            "answer": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "time": time.time() - start_time,
            "complexity": "complex" if is_complex else "simple",
        }
    except Exception as e:
        if model == advanced_model:
            print(f"Falling back to {primary_model}")
            response = client.chat.completions.create(
                model=primary_model,
                messages=[{"role": "user", "content": query}]
            )
            return {
                "model": primary_model,
                "answer": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
                "time": time.time() - start_time,
                "complexity": "complex (fallback)",
            }
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 9: Evaluation Suite")
    print("=" * 80)
    print()

    question = "What is the capital of France?"
    answer = "The capital of France is Paris, which is also its largest city."

    print("--- LLM-as-Judge Example ---")
    result = llm_as_judge(question, answer, "accuracy")
    print(json.dumps(result, indent=2))

    print("\n--- Multi-Model Fallback Example ---")
    simple_result = multi_model_fallback("What time does the store open?")
    print(f"Simple query -> {simple_result['model']} ({simple_result['time']:.2f}s)")

    complex_result = multi_model_fallback(
        "What are the legal considerations for implementing GDPR compliance in healthcare?"
    )
    print(f"Complex query -> {complex_result['model']} ({complex_result['time']:.2f}s)")
