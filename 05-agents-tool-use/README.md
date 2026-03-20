# Module 05 — AI Agents & Tool Use

**Goal:** Build a research agent that can call tools (web search, calculator, file reader) in a loop, deciding on its own when to act and when to answer.

**Time:** ~2 days

---

## Setup & running

```bash
pip install openai python-dotenv rich httpx

python agent.py "What is the population of Brazil and how does it compare to Argentina's?"
```

---

## What you'll learn

- How LLM tool use / function calling works under the hood
- The request → function call → tool response → reply loop
- How to define tools with JSON schemas that the model understands
- The ReAct pattern (Reason + Act) and why it matters
- How to build an agent loop that runs autonomously until it has an answer
- When agents are overkill vs. when they're the right abstraction

---

## Concepts

### What is an AI agent?

An agent is an LLM that can **decide what to do next**. Instead of one prompt → one response, the model enters a loop:

```
User question
    ↓
LLM thinks → decides to call a tool
    ↓
Tool executes → result sent back to LLM
    ↓
LLM thinks → calls another tool (or answers)
    ↓
... repeats until it has enough info ...
    ↓
Final answer
```

The key insight: **the model chooses which tool to call and with what arguments.** You don't hardcode the flow — the LLM figures it out.

### Function calling with OpenAI

OpenAI's API has native function calling. You define tools as JSON schemas wrapped in a `{"type": "function", "function": {...}}` structure, and the model returns `tool_calls` when it wants to call one.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in São Paulo?"}]
)
```

When the model wants to use a tool, the response looks like:

```python
# response.choices[0].finish_reason == "tool_calls"
# response.choices[0].message.tool_calls == [
#   ToolCall(id="call_abc123", function=Function(name="web_search", arguments='{"query": "weather São Paulo today"}'))
# ]
```

### The agent loop

The core pattern is a `for` loop with a max iteration cap:

```python
for i in range(MAX_ITERATIONS):
    response = client.chat.completions.create(model=..., tools=..., messages=messages)
    message = response.choices[0].message

    if response.choices[0].finish_reason == "stop":
        # Model is done — extract final answer
        break

    # Model wants to call tool(s)
    for tool_call in message.tool_calls:
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        # Append tool result to messages and loop again
```

Each iteration:
1. Send the full conversation (including prior tool calls and results) to the model
2. If the model responds with `finish_reason == "stop"`, you're done
3. If it responds with `tool_calls`, execute each tool, append the results, and loop

### The ReAct pattern

ReAct = **Re**ason + **Act**. The model alternates between:
- **Thinking** about what it knows and what it needs
- **Acting** by calling a tool
- **Observing** the tool's result

The model does this naturally when you give it tools — it will reason in its response before deciding to call a tool. This makes the agent's behavior interpretable and debuggable.

### Designing good tools

Good tool definitions make or break an agent:

| Do | Don't |
|---|---|
| Clear, specific descriptions | Vague names like `do_stuff` |
| Constrained input schemas | Accept anything as a string |
| Return structured, concise results | Return raw HTML or huge payloads |
| Handle errors gracefully | Let tools crash silently |

The model can only call tools it understands. If your tool description is ambiguous, the model will misuse it.

### Safety: max iterations

Always cap the number of iterations. A runaway agent loop burns tokens and money:

```python
MAX_ITERATIONS = 10

for i in range(MAX_ITERATIONS):
    response = client.chat.completions.create(...)
    if response.choices[0].finish_reason == "stop":
        break
    # ... handle tool calls ...
else:
    print("Agent hit max iterations without finishing.")
```

---

## Project: Research Agent

Build a CLI agent that takes a question, uses tools to gather information, and produces a well-sourced answer.

### Tools to implement

1. **`web_search`** — Search the web using a free API (DuckDuckGo via `httpx`, no API key needed)
2. **`calculator`** — Evaluate mathematical expressions safely
3. **`read_file`** — Read a local file and return its contents (useful for analyzing data files)

### Requirements

```
- Define 3+ tools with proper JSON schemas for OpenAI function calling
- Implement the agent loop (send messages → handle tool_calls → send tool results → repeat)
- Cap iterations at 10 to prevent runaway loops
- Display each tool call in real-time so you can watch the agent "think"
- Extract and display the final answer when finish_reason == "stop"
- Handle tool errors gracefully (return error message to the model, don't crash)
- Support a --verbose flag to show full message history
```

### Starter code

```python
# agent.py
import os
import sys
import json
import math
from pathlib import Path

import httpx
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI()
console = Console()

MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a research assistant with access to tools.
Use the tools to find accurate, up-to-date information before answering.
Always verify facts with the tools rather than relying on your training data.
When you have enough information, provide a clear, well-structured answer.
Cite your sources when possible."""


# ── Tool definitions (OpenAI function-calling format) ─────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this to find facts, statistics, news, or any information you're not confident about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports basic arithmetic, exponents (**), sqrt, and common math functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '(42 * 1.15) + 100'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file. Use this to analyze data files, configs, or documents on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
]


# ── Tool implementations ─────────────────────────────────────────


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo's HTML page and extract results."""
    # TODO: use httpx to hit DuckDuckGo's search
    # Hint: GET "https://html.duckduckgo.com/html/" with params={"q": query}
    # Parse the response text for result snippets
    pass


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    # TODO: use a safe subset of Python to evaluate math
    # Hint: only allow math functions, no builtins
    # Return the result as a string, or an error message
    pass


def read_file(path: str) -> str:
    """Read a file and return its contents (truncated if too long)."""
    # TODO: read the file, truncate to ~2000 chars if needed
    # Return an error message if file not found
    pass


def execute_tool(name: str, arguments: str) -> str:
    """Route a tool call to the right function."""
    # TODO: parse the JSON arguments string, dispatch to the right function
    # Return the result as a string
    # Catch exceptions and return error messages
    pass


# ── Agent loop ────────────────────────────────────────────────────


def run_agent(query: str, verbose: bool = False) -> str:
    """
    Run the agent loop:
    1. Send the query to OpenAI with tools
    2. If the model calls a tool, execute it and send the result back
    3. Repeat until the model gives a final answer or we hit MAX_ITERATIONS
    4. Return the final answer text
    """
    # TODO: implement the full agent loop
    # - maintain a messages list (starting with system + user)
    # - use console.print() with rich to show tool calls as they happen
    # - handle finish_reason == "stop" vs "tool_calls"
    # - for each tool call, append a message with role="tool" and tool_call_id
    pass


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research Agent with Tool Use")
    parser.add_argument("query", help="Your research question")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full message history")
    args = parser.parse_args()

    console.print(Panel(f"[bold]Query:[/bold] {args.query}", title="🔬 Research Agent"))

    answer = run_agent(args.query, verbose=args.verbose)

    console.print(Panel(Markdown(answer), title="📝 Answer", border_style="green"))
```

### Your task

1. **`web_search()`** — Hit DuckDuckGo's HTML endpoint, parse out result titles and snippets
2. **`calculator()`** — Safely evaluate math with a restricted set of allowed names
3. **`read_file()`** — Read a local file, truncate long files, handle missing files
4. **`execute_tool()`** — Parse the JSON arguments string, dispatch tool calls, catch errors
5. **`run_agent()`** — The main agent loop: messages → tool_calls → tool results → repeat

### Hints

<details>
<summary>web_search — parsing DuckDuckGo</summary>

```python
resp = httpx.get("https://html.duckduckgo.com/html/", params={"q": query}, follow_redirects=True)
# The results are in <a class="result__a"> tags and <a class="result__snippet"> spans
# You can use simple string parsing or regex — no need for BeautifulSoup
# Extract the first 5 results' titles and snippets
```
</details>

<details>
<summary>calculator — safe eval</summary>

```python
allowed_names = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "log": math.log,
    "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
}
result = eval(expression, {"__builtins__": {}}, allowed_names)
```
</details>

<details>
<summary>run_agent — the loop structure</summary>

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query},
]

for i in range(MAX_ITERATIONS):
    response = client.chat.completions.create(
        model=MODEL, tools=TOOLS, messages=messages,
    )
    choice = response.choices[0]
    message = choice.message

    # Append assistant message to history
    messages.append(message)

    if choice.finish_reason == "stop":
        # Extract text from message.content
        break

    # Handle tool calls
    for tool_call in message.tool_calls:
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
```
</details>

---

## Stretch goals

- Add a **`summarize_url`** tool that fetches a URL and returns a summary (use httpx to fetch, truncate to 5000 chars)
- Implement **parallel tool calls** — the model can request multiple tools at once, handle them concurrently
- Add a **`save_note`** tool that writes findings to a local file, building up research notes
- Stream the agent's text responses in real-time using `client.chat.completions.create(stream=True)`
- Add **conversation memory** — let the user ask follow-up questions
- Implement a **planning step**: before acting, have the agent write out its plan and show it to the user

---

## Key questions to answer before moving on

1. What happens if a tool returns an error? How should the agent handle it?
2. Why is it important to cap the number of iterations?
3. How does the model decide which tool to use? What if multiple tools could work?
4. What's the difference between giving the model a tool vs. hardcoding a pipeline?
5. How would you test an agent? What makes agent testing harder than testing a simple prompt?

---

## Resources

- [OpenAI function calling docs](https://platform.openai.com/docs/guides/function-calling)
- [ReAct paper](https://arxiv.org/abs/2210.03629)
- [DuckDuckGo HTML API](https://html.duckduckgo.com/html/)
- [OpenAI cookbook — function calling](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

---

**When done:** Mark Module 05 as shipped in the root README, commit, and move to [Module 06](../06-evals/).
