# Module 15 — Model Context Protocol (MCP)

## Goal
Build a real MCP server that exposes the NovaCRM support agent's tools, resources, and prompts over the Model Context Protocol. Drive it from two clients: Claude Desktop (zero-code) and a Python script that wires the server into a Claude agent loop. After this module, "I've built an MCP server" stops being a buzzword on your résumé and becomes a thing in your GitHub.

## Time estimate
~1 day (4–6 focused hours).

## What you'll learn
- Why MCP exists and what problem it solves (the "M×N integration" problem).
- MCP architecture: hosts, clients, servers, transports.
- The three primitives: **tools**, **resources**, and **prompts** — and when to use each.
- The official `mcp` Python SDK (`FastMCP`) for building servers.
- The two transports you actually need: **stdio** (local) and **streamable HTTP** (remote).
- Debugging with **MCP Inspector**.
- Wiring an MCP server into an Anthropic agent loop using the Python `mcp` client.
- Distributing a server: `uvx`, `pipx`, Docker, Claude Desktop config.

## Why this module exists

You finished Module 14 by porting your NovaCRM capstone to LangChain + LangGraph. The tools — `search_knowledge_base`, `lookup_customer`, `lookup_order`, `create_ticket` — are still hardcoded inside one repo. If a teammate wants those tools in their agent, in Cursor, in Claude Desktop, or in some new IDE next quarter, they have to copy your code and re-wire it.

That is the **M×N integration problem**: M models/clients × N tool integrations = an exploding mess. Every shop solves it slightly differently, badly.

MCP is the answer: a standard JSON-RPC protocol that any client can speak to any server. Write the tool once, expose it as an MCP server, and it works in Claude Desktop, Cursor, Windsurf, VS Code, Zed, your own Python agent — anywhere a client speaks the protocol.

This is not optional knowledge in 2026. It's table stakes for AI Engineer roles, and most candidates haven't actually built one.

---

## Concepts

### 1. The architecture

```
┌────────────────────────┐         ┌──────────────────────┐
│  Host (Claude Desktop, │  spawns │  MCP Server          │
│  Cursor, your script)  │────────▶│  (your code)         │
│                        │         │                      │
│  ┌──────────────────┐  │  JSON-  │  exposes:            │
│  │ MCP Client       │◀─┼─RPC────▶│   • tools            │
│  │ (one per server) │  │  over   │   • resources        │
│  └──────────────────┘  │  stdio  │   • prompts          │
│                        │  or HTTP│                      │
└────────────────────────┘         └──────────────────────┘
```

- **Host:** the app the user is in (Claude Desktop, Cursor, your CLI).
- **Client:** a thin shim inside the host. One client per connected server.
- **Server:** your code. Owns the tools/resources/prompts, runs as a subprocess (stdio) or web service (HTTP).
- **Protocol:** JSON-RPC 2.0 with a defined handshake (initialize → capability negotiation → operation).

You almost never write the JSON-RPC by hand. The SDK does it.

### 2. The three primitives

MCP servers can expose three kinds of things. Knowing which to use is the main design decision.

| Primitive | Who decides to invoke it | Use it for | Analogue |
|---|---|---|---|
| **Tool** | The model | Side-effectful actions, computations, queries the model chooses | Function calling |
| **Resource** | The host (often the user, sometimes the model) | Read-only context the user/host wants in the chat | A file or URL |
| **Prompt** | The user (via UI) | Reusable, parametrized prompt templates surfaced as slash commands | Saved prompts |

Concretely:

- `create_ticket(...)` → **tool**. The model decides when to call it.
- `kb://articles/refunds` → **resource**. The user attaches it; the model reads it.
- `/triage <issue>` → **prompt**. The user invokes it from the host UI.

A common mistake is making everything a tool. Resources and prompts exist precisely so the *user* can stay in control of context and intent without the model having to "decide" to look something up.

### 3. Transports

Two that matter:

- **stdio** — server runs as a subprocess of the host. Stdin/stdout carry JSON-RPC. Used for local tools (Claude Desktop, Cursor on your laptop). Easy, no auth, no networking.
- **Streamable HTTP** — server runs as a web service. Used for remote/shared tools. Supports auth (OAuth 2.1), multiple concurrent clients. (Older docs reference "SSE transport" — that was an earlier remote transport, now superseded by streamable HTTP.)

You'll use stdio for this module. HTTP is a stretch goal.

### 4. The Python SDK — FastMCP

The official SDK ships a high-level `FastMCP` class that handles all the JSON-RPC plumbing. You decorate Python functions:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("novacrm")

@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID."""
    return {"order_id": order_id, "status": "shipped"}

@mcp.resource("kb://articles/{slug}")
def kb_article(slug: str) -> str:
    """Return the markdown body of a KB article."""
    return open(f"data/kb/{slug}.md").read()

@mcp.prompt()
def triage(issue: str) -> str:
    """Triage a customer issue into a category and severity."""
    return f"Triage this customer issue:\n\n{issue}\n\nReturn JSON: {{category, severity}}."

if __name__ == "__main__":
    mcp.run()  # defaults to stdio
```

That's it. Type hints become the JSON schema. Docstrings become the descriptions the model sees. Decorators register the primitives.

### 5. Debugging with MCP Inspector

You will write an MCP server, run it, and discover something doesn't work. The fastest way to diagnose is **MCP Inspector** — a web UI that connects to your server and lets you click "list tools", "call tool", "read resource", "get prompt" with full request/response visibility.

```bash
npx @modelcontextprotocol/inspector python server.py
```

It opens a browser tab. Use it before you touch any host integration. If it works in Inspector, the server is fine and the issue is in your client config.

### 6. Connecting to Claude Desktop

Claude Desktop reads `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) on startup. Add an entry:

```json
{
  "mcpServers": {
    "novacrm": {
      "command": "python",
      "args": ["/abs/path/to/15-mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop. Your tools appear in the "🔌" menu. Resources show up when you click the paperclip. Prompts appear as slash commands.

### 7. Connecting to a Python client

The other half of the protocol: writing a client. You'll write a script that:
1. Spawns your server over stdio.
2. Lists its tools.
3. Translates them into Anthropic tool schemas.
4. Runs an agent loop where Claude can call your MCP tools.

This is exactly what Claude Desktop does internally — you're just doing it in 80 lines of Python. Once you've written it, MCP stops being magic.

### 8. Distribution

Three common patterns:

- **`uvx <package>`** — fastest. Publish your server to PyPI; users add `"command": "uvx", "args": ["your-package"]` to their config. No clones, no venvs.
- **Docker** — for HTTP servers, complex deps, or air-gapped environments.
- **Repo + manual install** — for internal/private servers.

For this module you'll stop at "runs locally from this repo". Publishing is a stretch goal.

---

## Project: NovaCRM MCP server

Port the NovaCRM support agent's surface into a real MCP server. Then drive it from both Claude Desktop and a Python agent client.

### Requirements

1. **Server (`server.py`)** exposing:
   - **Tools:** `search_knowledge_base`, `lookup_customer`, `lookup_order`, `create_ticket`.
   - **Resources:** every file in `data/kb/` exposed as `kb://articles/{slug}`, plus a dynamic resource template `customer://profile/{customer_id}`.
   - **Prompts:** `triage` (categorize a customer issue), `draft_response` (draft a customer-facing reply given an issue + retrieved context).
2. **Python client (`client.py`)** that:
   - Spawns the server over stdio.
   - Lists tools and converts them to Anthropic tool schemas.
   - Runs an agent loop with Claude, letting it call MCP tools until the model returns `end_turn`.
   - Prints each tool call + result so you can see what the model is doing.
3. **Claude Desktop integration:** a working `claude_desktop_config.example.json` and a screenshot or transcript of the tools being used inside Claude Desktop.
4. **Inspector check:** verify every tool/resource/prompt works in MCP Inspector before integrating.

### Starter files

- `server.py` — `FastMCP` server with stubs for every tool, resource, and prompt. TODOs mark each.
- `client.py` — Python agent client with stubs for the stdio handshake, tool listing, schema conversion, and the agent loop.
- `data/kb/*.md` — three sample KB articles (`billing.md`, `refunds.md`, `account.md`). Real-ish content; treat as the corpus.
- `data/customers.json` — a tiny fake customer/order database. Toy data, but matches the shape of a real CRM.
- `claude_desktop_config.example.json` — copy/edit this and drop into Claude Desktop's config dir.
- `requirements.txt` — `mcp`, `anthropic`, `python-dotenv`.
- `.env.example` — `ANTHROPIC_API_KEY`.

### Your task (in this order)

1. **Setup.** Create a venv, `pip install -r requirements.txt`, copy `.env.example` to `.env` and fill in the key.
2. **Implement the four tools** in `server.py`. Use the data in `data/customers.json` and `data/kb/`. Keep them simple — string match search is fine, this isn't a RAG module.
3. **Run the server through Inspector.** `npx @modelcontextprotocol/inspector python server.py`. Click each tool, confirm the schema looks right, call them with valid inputs.
4. **Implement the resources.** Both the static KB articles and the dynamic customer profile template. Re-check in Inspector.
5. **Implement the two prompts.** Re-check in Inspector — prompts have their own tab.
6. **Wire it into Claude Desktop.** Edit your real `claude_desktop_config.json`, restart Claude Desktop, and use one of your tools in a conversation.
7. **Implement the Python client.** Spawn the server, list tools, convert schemas, run the loop. Test with a query like *"Look up customer C-1001 and tell me their open orders."*
8. **Write a 5-line note** at the bottom of this README under "What I learned" — what you'd do differently, what surprised you. Future-you will thank you.

### Hints

<details>
<summary>Hint: how does FastMCP know the JSON schema of my tool?</summary>

It reads the function's type hints and docstring. Use real types (`str`, `int`, `list[str]`, `dict`, or Pydantic models for nested shapes) and write a one-line docstring describing what the tool does. The model only sees the docstring + schema, not your implementation, so the docstring is your prompt for the tool.
</details>

<details>
<summary>Hint: how do resource templates work?</summary>

`@mcp.resource("kb://articles/{slug}")` registers a *template*. The decorated function receives `slug` as an argument. To advertise concrete instances (so they show up in the host's resource picker), you also implement `list_resources` — or with FastMCP, you can register both a list function and the template separately. Read the SDK README's "Resources" section.
</details>

<details>
<summary>Hint: stdio vs HTTP — which one am I running?</summary>

`mcp.run()` with no arguments runs stdio. `mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)` runs HTTP. For this module: stdio. If you're seeing "no output" when running `python server.py` directly, that's correct — stdio servers wait for a client to connect over stdin.
</details>

<details>
<summary>Hint: converting MCP tool schemas to Anthropic tool schemas</summary>

`session.list_tools()` returns objects with `.name`, `.description`, `.inputSchema`. Anthropic's tool format wants `name`, `description`, `input_schema`. The schemas are both JSON Schema, so it's mostly a key rename. Build a list of dicts and pass it to `client.messages.create(tools=...)`.
</details>

<details>
<summary>Hint: the agent loop shape</summary>

```
messages = [{"role": "user", "content": user_input}]
while True:
    response = anthropic.messages.create(model=..., tools=mcp_tools, messages=messages)
    if response.stop_reason == "end_turn":
        return response
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = await session.call_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.content[0].text,  # adjust for content types
            })
    messages.append({"role": "user", "content": tool_results})
```

This is the same loop you wrote in Module 05; the only new bit is `await session.call_tool(...)` instead of a local Python function call.
</details>

<details>
<summary>Hint: my server crashes silently when Claude Desktop launches it</summary>

Claude Desktop swallows stderr by default. To debug:
1. Run the exact command from your config (`python /abs/path/to/server.py`) in a terminal first — does it import cleanly?
2. Check `~/Library/Logs/Claude/mcp*.log` on macOS.
3. Use absolute paths in the config. Relative paths break because Claude Desktop's CWD is not your project dir.
4. If you used `uv` or a venv, point `command` at that venv's Python (`/abs/path/to/.venv/bin/python`), not system Python.
</details>

## Stretch goals

- **Streamable HTTP transport.** Add a `--http` flag that flips `mcp.run()` to streamable-http and exposes the server on port 8000. Connect to it from `client.py`.
- **Publish to PyPI.** Add a `pyproject.toml`, `pip install build && python -m build`, push to TestPyPI, install with `uvx --from <test-pypi-url> novacrm-mcp`.
- **Wire it into a second client.** Cursor, Windsurf, Zed, or VS Code Copilot — pick one, add the server to its MCP config, confirm it works. The whole point of MCP is portability; prove it.
- **Auth.** Put the HTTP server behind a static bearer token. (OAuth is overkill for this module.)
- **Tool annotations.** Mark `create_ticket` with `destructiveHint=True` and observe how Claude Desktop surfaces the confirmation UI.

## Key questions

1. When would you expose something as a **resource** instead of a **tool**? Give a concrete example from NovaCRM.
2. Why does MCP use JSON-RPC instead of plain REST?
3. Stdio servers can't be shared across users. What changes when you move to streamable HTTP — protocol-wise, security-wise, deployment-wise?
4. Your server crashes mid-tool-call. What does the client see? What should a well-behaved server do instead?
5. What does the host (Claude Desktop) do that the protocol does *not* mandate? (Hint: think about consent UI, tool gating, resource pickers.)

## Resources

- MCP spec — https://modelcontextprotocol.io/specification
- Python SDK — https://github.com/modelcontextprotocol/python-sdk
- MCP Inspector — https://github.com/modelcontextprotocol/inspector
- Anthropic announcement — https://www.anthropic.com/news/model-context-protocol
- Server registry / examples — https://github.com/modelcontextprotocol/servers

---

## What I learned

<!-- Fill this in after you finish. Five lines, honest. -->
