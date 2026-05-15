"""NovaCRM MCP server.

Exposes the NovaCRM support agent's surface as Model Context Protocol primitives:
  - tools: search_knowledge_base, lookup_customer, lookup_order, create_ticket
  - resources: kb://articles/{slug}, customer://profile/{customer_id}
  - prompts: triage, draft_response

Run directly for stdio (this is what Claude Desktop does):
    python server.py

Run through MCP Inspector to debug:
    npx @modelcontextprotocol/inspector python server.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent
KB_DIR = ROOT / "data" / "kb"
CUSTOMERS_FILE = ROOT / "data" / "customers.json"
TICKETS_FILE = ROOT / "data" / "tickets.json"

mcp = FastMCP("novacrm")


def _load_customers() -> dict:
    return json.loads(CUSTOMERS_FILE.read_text())


def _load_tickets() -> list[dict]:
    if not TICKETS_FILE.exists():
        return []
    return json.loads(TICKETS_FILE.read_text())


def _save_tickets(tickets: list[dict]) -> None:
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2))

def _read_article(path: Path) -> dict:
    """Read a KB markdown file and extract its slug, title, and body."""
    body = path.read_text()
    # first markdown H1 — strip the leading "# " and any trailing whitespace
    title = next(
        (line[2:].strip() for line in body.splitlines() if line.startswith("# ")),
        path.stem,  # fallback if no H1
    )
    return {"slug": path.stem, "title": title, "body": body}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_knowledge_base(query: str, limit: int = 3) -> list[dict]:
    """Search NovaCRM knowledge base articles. Returns the top matches as
    {slug, title, snippet} dicts. Use this before answering policy or how-to
    questions so the response is grounded in the KB.
    """
    q = query.lower()
    articles = [_read_article(p) for p in sorted(KB_DIR.glob("*.md"))]

    scored = []
    for article in articles:
        haystack = (article["title"] + "\n" + article["body"]).lower()
        score = haystack.count(q)
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda pair: pair[0], reverse=True)


    results = []
    for score, article in scored[:limit]:
        body_lower = article["body"].lower()
        match_index = body_lower.find(q)

        if match_index == -1:
            # query hit the title but not the body — snippet from the top
            snippet = article["body"][:200]
        else:
            start = max(0, match_index - 100)
            end = match_index + 100
            snippet = article["body"][start:end]

        results.append({
            "slug": article["slug"],
            "title": article["title"],
            "snippet": snippet.strip(),
        })
    
    return results


@mcp.tool()
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer by ID (e.g. "C-1001"). Returns their profile and
    a summary of their orders. Returns {"error": "..."} if not found.
    """
    data = _load_customers()
    customer = next(
        (c for c in data["customers"] if c["id"] == customer_id),
        None,  # default if no match found
    )

    if customer is None:
        return {"error": f"customer {customer_id!r} not found"}

    orders_summary = [
        {"order_id": order["id"], "status": order["status"]}
        for order in customer["orders"]
    ]

    return {**customer, "orders_summary": orders_summary}


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID (e.g. "O-5001"). Returns the order record
    including items, total, status, and the owning customer_id.
    """
    # TODO: load customers.json, search every customer's orders for one
    # matching `order_id`, return it (with customer_id attached). Return
    # {"error": "..."} if not found.
    raise NotImplementedError


@mcp.tool()
def create_ticket(
    customer_id: str,
    subject: str,
    body: str,
    priority: str = "normal",
) -> dict:
    """Create a support ticket for a customer. `priority` is one of
    "low", "normal", "high", "urgent". Returns the created ticket record
    including a generated ticket_id.
    """
    # TODO:
    #   - validate priority against the allowed set
    #   - confirm customer_id exists (use lookup_customer or load directly)
    #   - generate a ticket_id like "T-<short-uuid>"
    #   - append a ticket dict to tickets.json with created_at = now (UTC ISO)
    #   - return the ticket dict
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("kb://articles/{slug}")
def kb_article(slug: str) -> str:
    """Return the markdown body of a KB article by slug."""
    # TODO: read KB_DIR / f"{slug}.md" and return its text. If the file does
    # not exist, raise a clear error — FastMCP will surface it to the client.
    raise NotImplementedError


@mcp.resource("customer://profile/{customer_id}")
def customer_profile(customer_id: str) -> str:
    """Return a markdown-formatted customer profile (name, email, plan,
    open ticket count, recent orders) for the given customer_id.
    """
    # TODO: load the customer, format a small markdown doc summarizing the
    # profile. This is what the *user* would attach to a chat — keep it
    # human-readable, not JSON.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def triage(issue: str) -> str:
    """Classify a customer issue into a category and severity. Surfaces in
    Claude Desktop as a slash command the user can invoke.
    """
    # TODO: return a prompt string that asks the model to:
    #   - categorize `issue` as one of: billing, account, technical, other
    #   - assign severity: low | normal | high | urgent
    #   - return strict JSON: {"category": "...", "severity": "...", "reason": "..."}
    raise NotImplementedError


@mcp.prompt()
def draft_response(issue: str, kb_context: str = "") -> str:
    """Draft a customer-facing reply given the issue and optional KB context."""
    # TODO: return a prompt string that instructs the model to draft a
    # concise, friendly reply. If `kb_context` is provided, the model must
    # ground the response in it and cite the relevant slug. If not, ask the
    # model to flag what info is missing rather than make things up.
    raise NotImplementedError


if __name__ == "__main__":
    mcp.run()
