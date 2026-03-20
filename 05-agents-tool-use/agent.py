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
    import re

    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    )

    results = []
    for match in re.finditer(
        r'<a rel="nofollow" class="result__a"[^>]*>(.+?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.+?)</a>',
        resp.text,
        re.DOTALL,
    ):
        title = re.sub(r"<.*?>", "", match.group(1)).strip()
        snippet = re.sub(r"<.*?>", "", match.group(2)).strip()
        results.append(f"- {title}: {snippet}")
        if len(results) >= 5:
            break
    return "\n".join(results) if results else "No results found."


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    """Read a file and return its contents (truncated if too long)."""
    try:
        text = Path(path).read_text()
        if len(text) > 2000:
            return text[:2000] + "n... (truncated)"
        return text
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

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
