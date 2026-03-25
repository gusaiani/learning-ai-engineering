# agent.py
import os
import sys
import json
import math
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv(Path(__file__).parent.parent / ".env")

client = Anthropic()
console = Console()

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a research assistant with access to tools.
Use the tools to find accurate, up-to-date information before answering.
Always verify facts with the tools rather than relying on your training data.
When you have enough information, provide a clear, well-structured answer.
Cite your sources when possible."""


# ── Tool definitions ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information. Use this to find facts, statistics, news, or any information you're not confident about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic, exponents (**), sqrt, and common math functions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '(42 * 1.15) + 100'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a local file. Use this to analyze data files, configs, or documents on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
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


def execute_tool(name: str, input_data: dict) -> str:
    """Route a tool call to the right function."""
    # TODO: dispatch to the right function based on name
    # Return the result as a string
    # Catch exceptions and return error messages
    pass


# ── Agent loop ────────────────────────────────────────────────────


def run_agent(query: str, verbose: bool = False) -> str:
    """
    Run the agent loop:
    1. Send the query to Claude with tools
    2. If Claude calls a tool, execute it and send the result back
    3. Repeat until Claude gives a final answer or we hit MAX_ITERATIONS
    4. Return the final answer text
    """
    # TODO: implement the full agent loop
    # - maintain a messages list
    # - use console.print() with rich to show tool calls as they happen
    # - handle stop_reason == "end_turn" vs "tool_use"
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
