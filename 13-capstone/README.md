# Module 13 — Capstone: AI Support Agent

**Goal:** Build and ship a production-grade AI customer support agent that combines everything from Modules 1–12 into one cohesive product. This is your portfolio centerpiece.

**Time:** ~4–5 days

---

## Setup & running

```bash
pip install openai python-dotenv fastapi uvicorn chromadb langfuse sse-starlette pydantic

# Ingest the sample knowledge base
python knowledge.py ingest sample_docs/

# Run the server
uvicorn server:app --reload --port 8000

# Chat (streaming)
curl -N http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How much does the Pro plan cost?"}'

# Chat with customer context
curl http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my subscription", "customer_id": "C-1001"}'

# Knowledge base contents
curl http://localhost:8000/knowledge

# Operational metrics
curl http://localhost:8000/metrics

# Run the eval suite
python evals.py
```

Langfuse tracing is optional. Add to `.env` if you want it:

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## What you'll learn

This module is different from the others. You already know how to build each piece. The challenge now is:

- How to combine RAG, agents, streaming, and observability into one system without them fighting each other
- How to design agent architectures that scale (router + specialist pattern)
- How to stream responses while an agent is mid-tool-loop
- How to trace a multi-agent request end-to-end with Langfuse
- How to eval a compound system (component evals vs. end-to-end)
- How to make real production decisions: what to cache, when to fail, how to degrade gracefully

---

## Architecture

```
Customer message (+ optional image, customer_id)
│
├── FastAPI receives request
│   └── Langfuse trace starts
│
├── Router Agent (gpt-4o-mini, structured output)
│   ├── Classifies: billing | technical | general | escalation
│   └── Picks the specialist
│
├── Specialist Agent (tool loop)
│   ├── search_knowledge_base(query)  →  RAG over company docs
│   ├── lookup_customer(id)           →  customer record
│   ├── lookup_order(id)              →  subscription / order info
│   ├── create_ticket(...)            →  escalation to human
│   └── analyze_image(path)           →  GPT-4o vision
│   │
│   └── Streams: status events during tool calls, tokens for final response
│
├── Response complete
│   └── Langfuse trace ends (cost, latency, tokens, tools used)
│
└── Metrics updated
```

### Module map

Every module you've completed has a role here:

| Component                                     | Module |
| --------------------------------------------- | ------ |
| API calls, system prompts, structured outputs | 01     |
| Prompt design for routing, RAG, tool use      | 02     |
| Document embeddings for knowledge base        | 03     |
| Retrieval pipeline (chunk → embed → search)   | 04     |
| Tool-calling agent loop                       | 05     |
| Eval suite (retrieval + response + routing)   | 06     |
| Streaming SSE, async FastAPI, error handling  | 07     |
| Design for future domain fine-tuning          | 08     |
| Router → specialist multi-agent pattern       | 09     |
| Langfuse tracing across agents                | 10     |
| Customer screenshot analysis via vision       | 11     |
| Overall architecture decisions                | 12     |

---

## Concepts

### From pieces to product

You've built a chat completion. You've built RAG. You've built an agent. You've built a streaming API. Individually, each took about two days.

Combining them takes four or five. Not because the code is 2.5× more, but because of the **integration tax**: every connection point between systems introduces decisions, failure modes, and trade-offs that don't exist in isolation.

Examples:

- Your agent calls `search_knowledge_base` as a tool. That tool runs a RAG pipeline. The RAG pipeline calls the embedding API. If the embedding API times out, does the agent retry? Fall back to keyword search? Tell the user? Each answer is reasonable — you have to decide.
- You're streaming the agent's final response. But the agent made three tool calls first, each taking 1–2 seconds. Do you show the user a loading spinner for 5 seconds, then stream? Or stream status messages during tool calls? The second is much better UX but requires a more complex event protocol.
- You want Langfuse to trace the full request. But the request spans a router agent, a specialist agent, two tool calls, and a RAG search. That's six spans nested inside one trace. Getting the span hierarchy right means threading trace context through every function call.

The integration tax is real, but it's also where senior engineers earn their keep. A junior can build each piece. A senior can make them work together.

### Router + specialist architecture

The simplest agent design: one agent, one system prompt, all tools. It works for small systems but degrades as complexity grows:

- More tools → longer tool descriptions → more prompt tokens → higher cost and worse tool selection
- One system prompt tries to cover billing tone AND technical debugging AND escalation policy → mediocre at all three
- Hard to eval: was the failure in routing, retrieval, or response generation?

The **router + specialist** pattern splits the work:

```
Router Agent (lightweight, fast, no tools)
├── "How much does Pro cost?"         → billing_agent
├── "API returns 429 error"           → technical_agent
├── "I hate your product"             → escalation (→ human handoff)
└── "Where are you based?"            → general_agent
```

The router is a cheap classification call — `gpt-4o-mini` with structured output, no tools. Each specialist gets a focused system prompt and only the tools it needs:

- **billing_agent**: search KB, lookup customer, lookup order, create ticket
- **technical_agent**: search KB, lookup customer, analyze image, create ticket
- **general_agent**: search KB only
- **escalation**: search KB, create ticket

Benefits:

1. **Cheaper.** The router call is ~100 tokens. Specialists only load relevant tool descriptions.
2. **Better quality.** Focused system prompts outperform swiss-army-knife prompts.
3. **Debuggable.** You can eval routing independently from response quality.
4. **Extensible.** Adding a new specialist doesn't touch existing ones.

The trade-off: sometimes a query spans categories ("I got an API error and I want a refund"). Handle this by letting the router pick the primary category and giving specialists enough context to address secondary concerns. Perfect routing isn't the goal — good-enough routing with graceful handling of edge cases is.

### Streaming with agent loops

In Module 07 you streamed a single chat completion. Clean and simple: tokens flow in, you forward them out.

With agents, it's messier. A typical flow:

```
User: "Can you check my billing status?"
Agent thinks → calls lookup_customer("C-1001")     [1.2s]
Agent thinks → calls lookup_order("ORD-5001")      [0.8s]
Agent thinks → generates response                  [streams ~2s]
```

Total: 4 seconds. Without streaming, the user sees nothing for 4 seconds. With naive streaming (only stream the final response), they see nothing for 2 seconds, then the text starts. With **event streaming**, you can do this:

```
event: status
data: {"tool": "lookup_customer", "status": "calling"}

event: status
data: {"tool": "lookup_customer", "status": "done"}

event: status
data: {"tool": "lookup_order", "status": "calling"}

event: status
data: {"tool": "lookup_order", "status": "done"}

event: token
data: {"content": "Based"}

event: token
data: {"content": " on"}

event: token
data: {"content": " your"}
...

event: done
data: {"cost": 0.0012, "tools_called": 2}
```

This requires your agent loop to be a **generator** that yields events — tool call starts, tool call completions, and response tokens. The FastAPI endpoint converts these events to SSE. The key design decision: the agent loop doesn't know it's being streamed. It just yields events. The server decides what to do with them.

### Observability across agents

In Module 10 you traced single-endpoint requests. The capstone has multi-step flows:

```
Trace: POST /chat
├── Span: router (82ms, gpt-4o-mini)
│   └── classification: "billing"
├── Span: billing_agent (3,400ms)
│   ├── Span: tool:search_knowledge_base (890ms)
│   │   ├── Span: embedding (120ms)
│   │   └── Span: chroma_query (45ms)
│   ├── Span: tool:lookup_customer (12ms)
│   ├── Span: llm_call (1,800ms, gpt-4o-mini)
│   │   ├── input_tokens: 1,247
│   │   ├── output_tokens: 312
│   │   └── cost: $0.0004
│   └── tools_called: 2
└── Total: 3,482ms | Cost: $0.0006
```

The pattern: **every function that does meaningful work gets its own span**, and spans nest to show the call hierarchy. Langfuse's `@observe` decorator handles this if you use it consistently:

```python
from langfuse.decorators import observe

@observe(name="route_query")
def route_query(message: str) -> str:
    ...

@observe(name="search_knowledge_base")
def search(query: str) -> list[dict]:
    ...
```

What to track on each span:

- **LLM calls**: model, input/output tokens, cost, the actual prompt and response
- **Tool calls**: tool name, arguments, result (truncated), latency
- **Errors**: the full exception, not just "failed"

### Eval strategy for compound systems

In Module 06 you built evals for RAG — retrieval precision and answer quality. For a compound system, you need **layered evals**:

**Layer 1 — Component evals** (fast, cheap, run often)

- Retrieval: given a question, are the top-k chunks relevant?
- Routing: given a message, does the router pick the right specialist?
- Tool selection: given a scenario, does the agent call the right tool?

**Layer 2 — Integration evals** (moderate cost)

- RAG + agent: does the agent use retrieved context correctly?
- Router + specialist: does the routed specialist give a better answer than a generic agent?

**Layer 3 — End-to-end evals** (expensive, run less often)

- Given a customer message, is the final response correct, helpful, and grounded?
- LLM-as-judge with a rubric: accuracy (0–5), helpfulness (0–5), hallucination (yes/no)

Why layers matter: if end-to-end evals fail, you need to know **where** the failure is. "The answer was wrong" could mean retrieval missed the right doc, the router picked the wrong specialist, or the specialist hallucinated. Layered evals pinpoint the breakage.

### Fine-tuning: when to reach for it

Module 08 covered fine-tuning mechanics. In this project, you won't fine-tune — but you should know when you would.

Signs that fine-tuning would help this system:

- The router misclassifies edge cases consistently, and no amount of prompt engineering fixes it → fine-tune a small classifier
- Specialists produce a wrong tone or format despite detailed prompts → fine-tune on example conversations
- Embedding search returns irrelevant chunks for domain-specific jargon → fine-tune the embedding model

Signs that fine-tuning is NOT the answer:

- The knowledge base is missing information → add docs, don't fine-tune
- The agent calls the wrong tool → fix the tool descriptions
- Answers are correct but verbose → adjust the system prompt

The capstone's architecture should make fine-tuning easy to slot in later: swap a model string, swap an embedding function — no structural changes.

---

## Project: AI Support Agent for NovaCRM

A production-ready API that handles customer support queries for NovaCRM, a fictional CRM platform. The agent searches a knowledge base, looks up customer data, processes actions, and streams responses — all with full observability.

### Requirements

1. **Knowledge base** (`knowledge.py`)
   - Ingest markdown files: read → chunk with overlap → embed via OpenAI → store in ChromaDB
   - Search: embed a query → find top-k similar chunks → return with scores and source metadata
   - List: show what's in the knowledge base (chunk count, source files)

2. **Agent system** (`agents.py`)
   - **Router**: classify incoming message → pick specialist using structured output
   - **Specialist agents**: focused system prompt + relevant tools per category
   - **Tool loop**: call tools until the agent has enough info, then respond
   - **Tools**: search_knowledge_base, lookup_customer, lookup_order, create_ticket, analyze_image
   - **Streaming**: yield events (tool status, response tokens) as a generator

3. **API server** (`server.py`)
   - `POST /chat` — streaming SSE endpoint
   - `POST /chat/sync` — non-streaming, returns complete response as JSON
   - `POST /ingest` — ingest a document into the knowledge base
   - `GET /knowledge` — list knowledge base contents
   - `GET /metrics` — operational stats (requests, latency, cost, errors)
   - `GET /health` — health check

4. **Eval suite** (`evals.py`)
   - Retrieval evals: do we find the right docs?
   - Routing evals: does the router classify correctly?
   - Response evals: LLM-as-judge scoring accuracy, helpfulness, hallucination
   - Summary report with pass rates and scores

5. **Observability** (`config.py` + decorators)
   - Langfuse tracing (graceful degradation when not configured)
   - Cost tracking per request
   - Token usage tracking

### Starter files

| File           | What's provided                                                             | What you implement                                |
| -------------- | --------------------------------------------------------------------------- | ------------------------------------------------- |
| `config.py`    | Clients, model constants, pricing table, ChromaDB                           | Langfuse init with graceful fallback              |
| `knowledge.py` | Collection setup, CLI dispatcher, `ingest_directory()`                      | Chunking, embedding, ingestion, search, listing   |
| `agents.py`    | Tool schemas, mock data, Pydantic models, system prompts, specialist config | Tool execution, agent loop, router, full pipeline |
| `server.py`    | FastAPI app, request/response models, metrics dict                          | All 6 endpoints                                   |
| `evals.py`     | Test cases, scoring models, CLI                                             | 4 eval functions + report generator               |
| `sample_docs/` | 3 NovaCRM knowledge base articles                                           | — (ready to ingest)                               |

### Your task

Work through the files in this order. Each phase can be tested independently.

**Phase 1 — Knowledge base** (`knowledge.py`)

1. Implement `chunk_text()` — split text into overlapping character-based chunks
2. Implement `embed_texts()` — call OpenAI embeddings API for a batch of texts
3. Implement `ingest_file()` — read → chunk → embed → upsert into ChromaDB
4. Implement `search()` — embed a query → ChromaDB query → return ranked results
5. Implement `list_documents()` — summarize knowledge base contents
6. Test: `python knowledge.py ingest sample_docs/ && python knowledge.py search "pricing"`

**Phase 2 — Agent** (`agents.py`)

7. Implement `execute_tool()` and all `_tool_*` handler functions
8. Implement `run_agent_loop()` — the core tool-calling loop that yields events
9. Implement `route_query()` — classify intent with structured output
10. Implement `run_support_agent()` — full pipeline: route → specialist → respond
11. Test: call `run_support_agent("How much is Pro?")` from a Python shell, iterate the events

**Phase 3 — API** (`server.py`)

12. Implement `POST /chat` — streaming SSE using the agent's event generator
13. Implement `POST /chat/sync` — collect all events, return final response
14. Implement the remaining endpoints: `/ingest`, `/knowledge`, `/metrics`, `/health`
15. Test: `uvicorn server:app --reload` → curl each endpoint

**Phase 4 — Evals** (`evals.py`)

16. Implement `eval_retrieval()` — check if the right docs come back
17. Implement `eval_routing()` — check if the router classifies correctly
18. Implement `eval_response()` — LLM-as-judge scoring of full pipeline output
19. Implement `run_eval_suite()` — run all cases, print a formatted report
20. Test: `python evals.py`

**Phase 5 — Observability** (`config.py` + decorators)

21. Initialize Langfuse in `config.py` with graceful degradation
22. Add `@observe` decorators to key functions: `route_query`, `run_agent_loop`, `search`, `embed_texts`
23. Verify traces appear in the Langfuse dashboard
24. Add cost tracking to the `done` event in the agent loop

---

### Hints

<details>
<summary>Hint for Phase 1 (knowledge base)</summary>

ChromaDB's `collection.upsert()` takes `ids`, `embeddings`, `documents`, and `metadatas` — all as lists. Use `upsert` instead of `add` so you can re-ingest the same file without errors.

For `collection.query()`, the result is nested: `{"ids": [[...]], "documents": [[...]], "distances": [[...]]}` — note the double brackets. Each inner list corresponds to one query. Since you're querying with one embedding, grab index `[0]` from each.

Cosine distance in ChromaDB: lower = more similar. A distance of 0 means identical, 2 means opposite.

</details>

<details>
<summary>Hint for Phase 2 (agent loop)</summary>

Streaming with tool calls is the hardest part. When the model wants to call a tool, the streamed chunks contain `delta.tool_calls` — but the tool call is split across multiple chunks. You need to accumulate:

```python
tool_calls_accumulator = {}
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {"id": tc.id, "name": tc.function.name, "arguments": ""}
            if tc.function.arguments:
                tool_calls_accumulator[idx]["arguments"] += tc.function.arguments
```

After the stream finishes, check if `tool_calls_accumulator` is non-empty. If so, it's Case A (tool calls). If the stream produced content instead, it's Case B (final response).

For the tool result message, the format is:

```python
{"role": "tool", "tool_call_id": tool_call_id, "content": result_string}
```

And the assistant message with tool calls:

```python
{"role": "assistant", "tool_calls": [
    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
    for tc in tool_calls_accumulator.values()
]}
```

</details>

<details>
<summary>Hint for Phase 3 (streaming SSE)</summary>

FastAPI's `StreamingResponse` accepts an async generator. Since `run_support_agent()` is synchronous, wrap it:

```python
async def event_generator():
    for event in run_support_agent(message, customer_id, image_path):
        yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"

return StreamingResponse(event_generator(), media_type="text/event-stream")
```

For the `/chat/sync` endpoint, just iterate the generator and collect events into lists — no streaming needed.

</details>

<details>
<summary>Hint for Phase 4 (evals)</summary>

For the LLM-as-judge in `eval_response()`, use `client.beta.chat.completions.parse()` with `response_format=ResponseScore`. The judge prompt should be neutral and specific:

```
You are evaluating an AI support agent's response.
Score accuracy (0-5): does the response contain correct factual information?
Score helpfulness (0-5): would a customer find this response useful?
Grounded (true/false): is all information in the response traceable to the knowledge base?
Hallucination (true/false): does the response state facts not present in the knowledge base?
```

Pass the customer message, agent response, reference answer, and source document name as context for the judge.

</details>

<details>
<summary>Hint for Phase 5 (Langfuse)</summary>

The simplest Langfuse integration uses the decorator approach:

```python
from langfuse.decorators import observe

@observe()
def route_query(message: str) -> RouteDecision:
    ...
```

Langfuse automatically creates nested spans when `@observe` functions call other `@observe` functions. So if `run_support_agent` (observed) calls `route_query` (observed) which calls `openai_client.chat.completions.create`, you get a proper trace hierarchy.

For graceful degradation when Langfuse isn't configured, you can define a no-op decorator:

```python
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
```

</details>

---

## Stretch goals

- **WebSocket chat** — add a `/ws` endpoint for a real-time bidirectional chat experience
- **Conversation memory** — maintain multi-turn context within a session (session ID → message history)
- **Semantic cache** — embed recent queries, return cached responses for semantically similar questions (saves cost on repeated queries)
- **Admin dashboard** — serve an HTML page from FastAPI with live metrics, recent conversations, and Langfuse trace links
- **Reranking** — add a cross-encoder reranking step after the initial vector search to improve retrieval precision
- **Batch ingestion API** — accept multiple files via multipart upload with a progress endpoint
- **Rate limiting** — per-customer rate limiting to prevent abuse
- **A/B eval** — compare router+specialist vs. single-agent on the same eval set. Measure quality AND cost differences.
- **Chat UI** — build a simple frontend (even plain HTML + fetch) that consumes the SSE stream and renders tool status + response tokens

---

## Key questions

1. The router adds an extra LLM call on every request. Under what conditions does this pay for itself? When would you skip the router and use a single agent instead?

2. Your RAG retrieval returns the right chunk, but the agent ignores it and answers from parametric knowledge (training data). How would you detect this? How would you fix it?

3. You notice the eval suite passes but real users complain about bad answers. What's the gap between your eval cases and real usage? How would you close it?

4. The `analyze_image` tool sends the full image to GPT-4o every time. If a customer sends the same screenshot in a follow-up message, you pay the vision cost again. How would you design a caching strategy for images?

5. You want to deploy this to production. List the top 5 things you'd change before real users hit it. Think about: secrets management, database for customer data, authentication, error recovery, and scaling.

6. If NovaCRM had 10,000 knowledge base articles instead of 3, what would break first in your pipeline? What's the fix?

---

## Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI Streaming](https://platform.openai.com/docs/api-reference/streaming)
- [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)
- [Langfuse Python Decorator](https://langfuse.com/docs/sdk/python/decorators)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Server-Sent Events spec](https://html.spec.whatwg.org/multipage/server-sent-events.html)
