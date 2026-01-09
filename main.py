import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("SUPER_MIND_API_KEY"),
    base_url="https://space.ai-builders.com/backend/v1"
)


def web_search(query: str) -> dict:
    """Call the internal search API to search the web."""
    url = "https://space.ai-builders.com/backend/v1/search/"
    headers = {
        "Authorization": f"Bearer {os.getenv('SUPER_MIND_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "keywords": [query],
        "max_results": 3
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


# Tool schema for LLM function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date information or facts about current events, people, places, or any topic you don't have information about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class ChatRequest(BaseModel):
    user_message: str


@app.get("/hello/{user_input}")
async def read_hello(user_input: str):
    return {"message": f"Hello, World {user_input}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    # Initialize conversation with user message
    messages = [
        {"role": "user", "content": request.user_message}
    ]

    max_turns = 3

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"[Turn {turn + 1}/{max_turns}]")
        print(f"{'='*60}")

        # Call LLM
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        # Check if LLM wants to call a tool
        if message.tool_calls:
            # Log tool decision
            for tool_call in message.tool_calls:
                print(f"\n[Agent] Decided to call tool: '{tool_call.function.name}'")
                print(f"[Agent] Arguments: {tool_call.function.arguments}")

                # Execute the tool
                import json
                args = json.loads(tool_call.function.arguments)

                if tool_call.function.name == "web_search":
                    tool_result = web_search(args["query"])
                    print(f"[System] Tool Output: {json.dumps(tool_result, indent=2)}")

                    # Add assistant message with tool call to history
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })

                    # Add tool result to history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(tool_result)
                    })
        else:
            # No tool calls - this is the final answer
            print(f"\n[Agent] Final Answer: {message.content}")
            print(f"{'='*60}\n")
            return {
                "content": message.content,
                "tool_calls": None
            }

    # If we've exhausted max_turns, return the last message
    print(f"\n[System] Reached maximum turns ({max_turns})")
    print(f"{'='*60}\n")
    return {
        "content": message.content or "I've reached the maximum number of tool calls. Please try rephrasing your question.",
        "tool_calls": None
    }

