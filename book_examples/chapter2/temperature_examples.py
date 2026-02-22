"""
Chapter 2: Temperature and Top-P - LLM Settings for Reliability

Demonstrates parameter tuning as a first-class reliability control.
Lower temperature = more deterministic = better for factual tasks.
"""

from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def temperature_example():
    """Demonstrates temperature parameter impact on reliability."""
    conversation = [{"role": "user", "content": "What are the specs for the iPhone 12?"}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=conversation,
    )
    print(f"Low temperature response (factual, consistent):\n{response.choices[0].message.content}\n")


def top_p_example():
    """Demonstrates top-p (nucleus) sampling parameter."""
    conversation = [{
        "role": "user",
        "content": "I'm looking for a new pair of wireless headphones for my daily commute. What do you suggest?"
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        top_p=0.95,
        messages=conversation,
    )
    print(f"Response with top_p=0.95:\n{response.choices[0].message.content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 2: Temperature and Top-P Examples")
    print("=" * 80)
    print()

    print("--- Temperature Example ---")
    temperature_example()

    print("\n--- Top-P Example ---")
    top_p_example()
