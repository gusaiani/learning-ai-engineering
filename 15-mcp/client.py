"""NovaCRM MCP client.

Spawns the local server.py over stdio, lists its tools, converts them to the
Anthropic tool schema, and runs an agent loop where Claude can call the MCP
tools until it returns end_turn.

Usage:
    python client.py "Look up customer C-1001 and summarize their orders."
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

MODEL = "claude-sonnet-4-6"
SERVER_PATH = Path(__file__).parent / "server.py"


def mcp_tools_to_anthropic(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions into Anthropic's tool schema.

    MCP tools have:  .name, .description, .inputSchema  (JSON Schema dict)
    Anthropic wants: name,  description,  input_schema (JSON Schema dict)
    """
    # TODO: build and return a list of {name, description, input_schema} dicts.
    raise NotImplementedError


async def run_agent(user_input: str) -> str:
    """Run one user turn through Claude with MCP tools available.

    Returns the final assistant text after the loop exits with end_turn.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # TODO: call session.list_tools(), then convert the .tools list
            # using mcp_tools_to_anthropic.
            tools = ...

            anthropic = Anthropic()
            messages = [{"role": "user", "content": user_input}]

            while True:
                # TODO:
                #   1. Call anthropic.messages.create with model=MODEL,
                #      tools=tools, messages=messages, max_tokens=2048.
                #   2. If response.stop_reason == "end_turn", break and
                #      return the concatenated text from response.content
                #      blocks of type "text".
                #   3. Otherwise, append the assistant turn to messages,
                #      then for every content block where block.type ==
                #      "tool_use":
                #         - print f"→ {block.name}({block.input})"
                #         - result = await session.call_tool(block.name, block.input)
                #         - extract result.content[0].text (handle non-text
                #           content types defensively if you want — for this
                #           module text is enough)
                #         - append a tool_result block with the matching
                #           tool_use_id
                #      Append the collected tool_results as a single user
                #      turn and loop again.
                raise NotImplementedError


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python client.py "<your question>"')
        sys.exit(1)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY missing — copy .env.example to .env and fill it in.")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    final = asyncio.run(run_agent(user_input))
    print("\n=== final ===")
    print(final)


if __name__ == "__main__":
    main()
