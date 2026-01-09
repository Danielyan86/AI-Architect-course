import os
import re
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("SUPER_MIND_API_KEY"),
    base_url="https://space.ai-builders.com/backend/v1"
)


def web_search(query: str) -> dict:
    """Call the internal search API to search the web."""
    try:
        url = "https://space.ai-builders.com/backend/v1/search/"
        headers = {
            "Authorization": f"Bearer {os.getenv('SUPER_MIND_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "keywords": [query],
            "max_results": 3
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        print(f"[Error] Search request timed out for query: {query}")
        return {"error": "Search request timed out", "query": query}
    except requests.exceptions.HTTPError as e:
        print(f"[Error] HTTP error during search: {e}")
        return {"error": f"HTTP error: {e.response.status_code}", "query": query}
    except requests.exceptions.RequestException as e:
        print(f"[Error] Request failed: {e}")
        return {"error": f"Request failed: {str(e)}", "query": query}
    except Exception as e:
        print(f"[Error] Unexpected error in web_search: {e}")
        return {"error": f"Unexpected error: {str(e)}", "query": query}


def read_page(url: str) -> dict:
    """Fetch a webpage and extract the main text content."""
    try:
        # Fetch the page
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Limit text length to avoid token overflow
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "... (truncated)"

        return {
            "url": url,
            "text": text,
            "length": len(text)
        }

    except requests.exceptions.Timeout:
        print(f"[Error] Page request timed out for URL: {url}")
        return {"error": "Page request timed out", "url": url}
    except requests.exceptions.HTTPError as e:
        print(f"[Error] HTTP error fetching page: {e}")
        return {"error": f"HTTP error: {e.response.status_code}", "url": url}
    except requests.exceptions.RequestException as e:
        print(f"[Error] Request failed: {e}")
        return {"error": f"Request failed: {str(e)}", "url": url}
    except Exception as e:
        print(f"[Error] Unexpected error in read_page: {e}")
        return {"error": f"Unexpected error: {str(e)}", "url": url}


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
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch and read the text content from a specific webpage URL. Use this when you need to read the detailed content of a page, such as documentation, articles, or changelogs. The function will extract the main text and remove scripts, styles, and navigation elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to read (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


class ChatRequest(BaseModel):
    user_message: str


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Root route to serve the frontend
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/hello/{user_input}")
async def read_hello(user_input: str):
    return {"message": f"Hello, World {user_input}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    # Initialize conversation with user message
    messages = [
        {"role": "user", "content": request.user_message}
    ]

    max_turns = 10

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
            import json

            # Add assistant message with ALL tool calls first (CRITICAL: do this once, not per tool)
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })

            # Then execute each tool and add results
            for tool_call in message.tool_calls:
                print(f"\n[Agent] Decided to call tool: '{tool_call.function.name}'")
                print(f"[Agent] Arguments: {tool_call.function.arguments}")

                # Execute the tool
                args = json.loads(tool_call.function.arguments)

                # Route to the appropriate tool
                if tool_call.function.name == "web_search":
                    tool_result = web_search(args["query"])
                elif tool_call.function.name == "read_page":
                    tool_result = read_page(args["url"])
                else:
                    tool_result = {"error": f"Unknown tool: {tool_call.function.name}"}

                print(f"[System] Tool Output: {json.dumps(tool_result, indent=2)}")

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

