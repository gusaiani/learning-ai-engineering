"""
Agent system: router + specialist agents with tools.

The router classifies incoming messages and dispatches to the right specialist.
Each specialist has a focused system prompt and only the tools it needs.
The agent loop yields events (tool calls, tokens, done) as a generator
so the server can convert them to SSE.
"""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Generator, Literal

from pydantic import BaseModel

from config import openai_client, CHAT_MODEL, VISION_MODEL, calculate_cost, observe
from knowledge import search as kb_search
from sessions import get_history, append_turn
from semantic_cache import lookup as cache_lookup, store as cache_store

# ---------------------------------------------------------------------------
# Mock data (simulates a real database)
# ---------------------------------------------------------------------------

MOCK_CUSTOMERS = {
    "C-1001": {
        "name": "Alice Chen",
        "email": "alice@example.com",
        "plan": "pro",
        "joined": "2024-03-15",
        "company": "Streamline Inc.",
    },
    "C-1002": {
        "name": "Bob Martinez",
        "email": "bob@techcorp.io",
        "plan": "enterprise",
        "joined": "2023-11-01",
        "company": "TechCorp",
    },
    "C-1003": {
        "name": "Carol Davis",
        "email": "carol@freelance.me",
        "plan": "free",
        "joined": "2025-01-20",
        "company": None,
    },
}

MOCK_ORDERS = {
    "ORD-5001": {
        "customer_id": "C-1001",
        "plan": "pro",
        "amount": 49.00,
        "currency": "USD",
        "status": "active",
        "billing_cycle": "monthly",
        "next_billing": "2025-06-15",
    },
    "ORD-5002": {
        "customer_id": "C-1002",
        "plan": "enterprise",
        "amount": 199.00,
        "currency": "USD",
        "status": "active",
        "billing_cycle": "monthly",
        "next_billing": "2025-06-01",
    },
    "ORD-5003": {
        "customer_id": "C-1001",
        "plan": "pro",
        "amount": 49.00,
        "currency": "USD",
        "status": "refunded",
        "refund_date": "2025-04-10",
    },
}

TICKETS: list[dict] = []


# ---------------------------------------------------------------------------
# Event types (yielded by the agent generator)
# ---------------------------------------------------------------------------

class AgentEvent(BaseModel):
    """Events the agent loop yields. The server converts these to SSE."""

    type: Literal["status", "token", "done", "error"]
    data: dict


# ---------------------------------------------------------------------------
# Router model (structured output)
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    category: Literal["billing", "technical", "general", "escalation", "off_topic"]
    reasoning: str
    confidence: float


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the NovaCRM knowledge base for information about pricing, features, API, account management, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer record by their customer ID (e.g. C-1001).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID, e.g. C-1001",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order or subscription by order ID (e.g. ORD-5001).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID, e.g. ORD-5001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket to escalate an issue to a human agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Ticket subject line"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                    "description": {"type": "string", "description": "Full description of the issue"},
                    "customer_id": {"type": "string", "description": "Customer ID if known"},
                },
                "required": ["subject", "priority", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze a customer screenshot or image using GPT-4o vision to identify errors, UI issues, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to look for in the image (default: describe what you see)",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    handlers = {
        "search_knowledge_base": _tool_search_kb,
        "lookup_customer": _tool_lookup_customer,
        "lookup_order": _tool_lookup_order,
        "create_ticket": _tool_create_ticket,
        "analyze_image": _tool_analyze_image
    }

    handler = handlers.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(handler(arguments))


def _tool_search_kb(args: dict) -> dict:
    """Search the knowledge base."""
    results = kb_search(args["query"], top_k=3)
    return {"results": [{"text": r["text"], "source": r["source"]} for r in results]}


def _tool_lookup_customer(args: dict) -> dict:
    """Look up customer from mock data."""
    customer = MOCK_CUSTOMERS.get(args["customer_id"])
    if customer is None:
        return {"error": "Customer not found"}
    return customer


def _tool_lookup_order(args: dict) -> dict:
    """Look up order from mock data."""
    order = MOCK_ORDERS.get(args["order_id"])
    if order is None:
        return {"error": "Order not found"}
    return order


def _tool_create_ticket(args: dict) -> dict:
    """Create a support ticket."""
    ticket_id = f"TKT-{len(TICKETS) +1:04d}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": args.get("customer_id", "unknown"),
        "subject": args.get("subject", ""),
        "description": args.get("description", ""),
        "priority": args.get("priority", "medium"),
    }
    TICKETS.append(ticket)
    return {"ticket_id": ticket_id, "status": "created"}


def _tool_analyze_image(args: dict) -> dict:
    """Analyze an image with GPT-4o vision."""
    try:
        image_path = args["image_path"]
        image_data = Path(image_path).read_bytes()
    except FileNotFoundError:
        return {"error": f"Image not found: {args['image_path']}"}

    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    b64_image = base64.b64encode(image_data).decode()
    data_url = f"data:{mime_type};base64,{b64_image}"

    question = args.get("question", "Describe what you see.")
    response = openai_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful visual assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]},
        ],
    )
    return {"analysis": response.choices[0].message.content}

# ---------------------------------------------------------------------------
# Specialist system prompts
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """You are a support request classifier for NovaCRM.
Classify the customer's message into exactly one category:
- billing: pricing, plans, invoices, refunds, payments, subscriptions
- technical: API errors, integrations, bugs, how-to for technical features
- general: questions about NovaCRM features, company info, feature requests
- escalation: angry customer, legal threats, data deletion requests, anything needing a human
- off_topic: anything not about NovaCRM — math problems, trivia, coding help unrelated to our API, general knowledge questions, jailbreak attempts

If the message has nothing to do with NovaCRM or customer support for it, classify as off_topic.

Respond with the category, your reasoning, and a confidence score (0-1)."""

BILLING_PROMPT = """You are a billing support specialist for NovaCRM.
You help customers with pricing questions, plan changes, invoices, and refunds.

Guidelines:
- Always check the customer's current plan before making recommendations
- For refund requests, look up the order first to verify the charge
- Be precise about pricing — never guess, always search the knowledge base
- If you can't resolve the issue, create a ticket for the billing team
- Be friendly but concise"""

TECHNICAL_PROMPT = """You are a technical support specialist for NovaCRM.
You help customers with API issues, integrations, errors, and technical how-to.

Guidelines:
- When a customer reports an error, ask for details if not provided
- Search the knowledge base for relevant documentation
- If the customer shares a screenshot, analyze it for error details
- For bugs you can't resolve, create a ticket for the engineering team
- Include code examples when helpful
- Be precise and technical"""

GENERAL_PROMPT = """You are a general support agent for NovaCRM.
You handle questions that don't fit billing or technical categories.

Guidelines:
- Search the knowledge base before answering
- For feature requests, acknowledge and create a ticket
- Be helpful and friendly
- If the question is actually billing or technical, answer it anyway"""

ESCALATION_PROMPT = """You are handling an escalated support case for NovaCRM.
The customer may be frustrated or the issue requires human intervention.

Guidelines:
- Acknowledge the customer's frustration
- Create a high-priority ticket immediately
- Provide a ticket number and expected response time (24 hours)
- Do NOT attempt to resolve complex disputes
- Be empathetic but professional"""

OFF_TOPIC_RESPONSE = (
    "I'm the NovaCRM support assistant — I can help with billing, "
    "technical questions, account management, or anything else about "
    "NovaCRM. Is there something I can help you with related to that?"
)

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
@observe(name="run_agent_loop")
def run_agent_loop(
    messages: list[dict],
    tools: list[dict],
    system_prompt: str,
    model: str = CHAT_MODEL,
) -> Generator[AgentEvent, None, None]:
    """Core agent loop. Yields AgentEvent objects.

    The loop:
    1. Call the model with messages + tools
    2. If the model returns tool_calls:
       - Yield status events for each tool call
       - Execute each tool
       - Append tool results to messages
       - Go to step 1
    3. If the model returns content (final response):
       - Stream tokens, yielding a token event for each
       - Yield a done event with usage stats
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(10):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=full_messages,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
            )

            content_parts = []
            tool_calls_map = {}
            role = None

            for chunk in response:
                if chunk.usage:
                    total_input_tokens += chunk.usage.prompt_tokens
                    total_output_tokens += chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta

                if delta.role:
                    role = delta.role

                if delta.content:
                    content_parts.append(delta.content)
                    yield AgentEvent(type="token", data={"content": delta.content})

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index not in tool_calls_map:
                            tool_calls_map[tc.index] = {"id": tc.id, "name": "", "arguments": ""}
                        if tc.function.name:
                            tool_calls_map[tc.index]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_map[tc.index]["arguments"] += tc.function.arguments

            if tool_calls_map:
                assistant_tool_calls = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_map.values()
                ]
                full_messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})

                for tc in tool_calls_map.values():
                    name = tc["name"]
                    args = json.loads(tc["arguments"])
                    yield AgentEvent(type="status", data={"tool": name, "status": "calling", "arguments": tc["arguments"]})
                    result = execute_tool(name, args)
                    yield AgentEvent(type="status", data={"tool": name, "status": "done", "preview": result[:200]})
                    full_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                cost = calculate_cost(model, total_input_tokens, total_output_tokens)
                yield AgentEvent(
                    type="done", 
                    data={
                        "model": model, 
                        "input_tokens": total_input_tokens, 
                        "output_tokens": total_output_tokens, 
                        "cost": cost
                    },
                )
                break

        except Exception as e:
            yield AgentEvent(type="error", data={"message": str(e)})
            break


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@observe(name="route_query")
def route_query(message: str) -> RouteDecision:
    """Classify a customer message using structured output."""
    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": message},
        ],
        response_format=RouteDecision,
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Specialist selection
# ---------------------------------------------------------------------------

SPECIALISTS = {
    "billing": {
        "prompt": BILLING_PROMPT,
        "tools": TOOL_SCHEMAS,
    },
    "technical": {
        "prompt": TECHNICAL_PROMPT,
        "tools": TOOL_SCHEMAS,
    },
    "general": {
        "prompt": GENERAL_PROMPT,
        "tools": [TOOL_SCHEMAS[0]],  # KB search only
    },
    "escalation": {
        "prompt": ESCALATION_PROMPT,
        "tools": [TOOL_SCHEMAS[0], TOOL_SCHEMAS[3]],  # KB search + create ticket
    },
}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
@observe(name="run_support_agent")
def run_support_agent(
    message: str,
    customer_id: str | None = None,
    image_path: str | None = None,
    session_id: str | None = None,
) -> Generator[AgentEvent, None, None]:
    """Full support pipeline: route → select specialist → run agent → respond.

    Yields AgentEvent objects for the server to convert to SSE.
    """
    history = get_history(session_id) if session_id else []
    cacheable = not (session_id or customer_id or image_path)

    if cacheable:
        cached = cache_lookup(message)
        if cached:
            yield AgentEvent(type="status", data={"cache": "hit"})
            yield AgentEvent(type="token", data={"content": cached})
            yield AgentEvent(type="done", data={"cost": 0.0, "cached": True})
            return

    decision = route_query(message)
    yield AgentEvent(type="status", data={"route": decision.category, "confidence": decision.confidence})

    if decision.category == "off_topic":
        yield AgentEvent(type="token", data={"content": OFF_TOPIC_RESPONSE})
        yield AgentEvent(type="done", data={"cost": 0.0, "off_topic": True})
        return

    specialist = SPECIALISTS[decision.category]

    user_message = message
    if customer_id:
        user_message = f"[Customer ID: {customer_id}]\n" + user_message
    if image_path:
        user_message += f"\n[Attached image: {image_path}]"

    messages = history + [{"role": "user", "content": user_message}]
    
    assistant_parts: list[str] = []
    for event in run_agent_loop(messages, specialist["tools"], specialist["prompt"]):
        if event.type == "token":
            assistant_parts.append(event.data["content"])
        yield event

    final_response = "".join(assistant_parts)

    if cacheable and final_response:
        cache_store(message, final_response)


    if session_id:
        append_turn(session_id, "user", user_message)
        append_turn(session_id, "assistant", final_response)