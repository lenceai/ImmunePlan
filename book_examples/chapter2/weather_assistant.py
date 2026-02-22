"""
Chapter 2: Weather Assistant - Function Calling for Reliability

Demonstrates the foundational pattern from the book:
  - Define function schema
  - Let model decide when to call
  - Retrieve real data
  - Compose response

This pattern separates language generation from fact retrieval,
which is a core reliability technique.
"""

from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_current_weather(location, unit="fahrenheit"):
    """Simulated weather function - in production this calls a real API."""
    weather_data = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_data)


def weather_assistant_example():
    """Demonstrates function calling with weather assistant."""

    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        }
    }]

    messages = [{"role": "user", "content": "What's the weather like in Boston?"}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"Model called function: {function_name}")
            print(f"With arguments: {function_args}")

            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "fahrenheit"),
                )

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        final_answer = second_response.choices[0].message.content
        print(f"\nFinal response:\n{final_answer}")
        return final_answer
    else:
        print(response_message.content)
        return response_message.content


if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 2: Weather Assistant with Function Calling")
    print("=" * 80)
    print()
    weather_assistant_example()
