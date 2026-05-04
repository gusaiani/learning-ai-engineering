# Framework Migration — From-Scratch Capstone → LangChain + LangGraph

## Goal
Port the from-scratch capstone (Module 13) to LangChain + LangGraph, stage by stage, so you can articulate exactly what these frameworks abstract — and what they don't — based on code you wrote yourself.

## Time estimate
~1 day (4 staged baby-step sessions, ~1–2 hours each).

## What you'll learn
- What `ChatPromptTemplate` actually buys you over f-strings + dicts.
- How LangChain wraps a vector store (Chroma) into a `Retriever` interface and what that interface assumes.
- How a LangGraph `StateGraph` expresses an agent loop as a typed state machine, and why the from-scratch loop in `agents.py` maps cleanly onto nodes + conditional edges.
- How LangChain's Langfuse `CallbackHandler` replaces hand-placed `@observe` decorators.
- Where the framework adds value (composition, streaming, callbacks) vs. where it adds weight (extra abstractions for code you already had).
- The migration patterns you'd use on a real codebase: keep boundaries (FastAPI, sessions, rate limiting) untouched, swap internals incrementally, never big-bang.

## Concepts

### LangChain vs. LangGraph
**LangChain** is a library of LLM building blocks: prompt templates, chat models, output parsers, vector stores, retrievers, and a "chain" abstraction (LCEL — LangChain Expression Language) that pipes them together with `|`. It's strongest for linear pipelines — prompt → model → parse — and for the standard interfaces it imposes on vector stores and retrievers.

**LangGraph** is a separate library (same org) for *stateful, branching* workflows. You declare a `TypedDict` state, register nodes (functions that read/write state), and add edges (including *conditional* edges that pick the next node based on state). It's the right tool for an agent loop — exactly what `run_agent_loop` in `agents.py` is doing by hand.

You'll use both: LangChain for the prompt + retriever + model layer, LangGraph for the agent loop.

### `ChatPromptTemplate`
Today, `agents.py` builds messages with string concatenation and dict literals:

```python
full_messages = [{"role": "system", "content": system_prompt}] + messages
```

`ChatPromptTemplate` lets you declare a template once and call it with variables:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", SHARED_SYSTEM_PREFIX + BILLING_PROMPT),
    ("placeholder", "{history}"),
    ("user", "{message}"),
])
messages = prompt.format_messages(history=history, message=user_message)
```

What you gain: the template is a first-class object you can compose into chains (`prompt | model | parser`), partial-apply, log, or swap. What you don't gain: nothing automatic — you're still on the hook for what goes in those slots.

### LangChain `Chroma` + `Retriever`
Today, `knowledge.py` calls Chroma directly: `collection.upsert(...)`, `collection.query(...)`, and you parse the result dict yourself.

LangChain wraps Chroma in a `VectorStore` with a uniform interface across all backends (Pinecone, Weaviate, pgvector, FAISS, …):

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="knowledge_base",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512),
    persist_directory=str(CHROMA_PATH),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("how do I get an API key?")
```

What you gain: one interface for any vector store — swap Chroma for Pinecone by changing one import. Built-in `Document` objects with `page_content` + `metadata`. What you trade: less direct control over the query payload; LangChain decides the schema.

### LangGraph `StateGraph`
Today, `run_agent_loop` is a `for _ in range(10):` loop with two branches: "model returned tool calls → execute and recurse" or "model returned content → stream and finish".

LangGraph expresses the same shape as a graph:

```
        ┌──────────┐
        │  agent   │  (calls the model with current messages)
        └────┬─────┘
             │ should_continue?
       ┌─────┴─────┐
       │           │
       ▼           ▼
 ┌──────────┐    END
 │  tools   │  (executes tool calls, appends results)
 └────┬─────┘
      └──────► back to agent
```

In code:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", run_tools)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

graph = builder.compile()
```

What you gain: the loop's *shape* becomes data — visualizable, testable per node, easy to add nodes (e.g. a guardrail step). Streaming via `graph.astream(...)`. What you trade: more files and types for what was a tight 80-line function.

### Langfuse via callback handler
Today: `@observe(name="run_agent_loop")` sprinkled across `agents.py` and `knowledge.py`.

With LangChain integration: add the Langfuse `CallbackHandler` to the chain/graph config and every model call, retrieval, and tool execution gets traced automatically — no decorators per function.

```python
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
graph.invoke(state, config={"callbacks": [langfuse_handler]})
```

### What we're NOT migrating (and why)
| File | Why it stays as-is |
|---|---|
| `server.py` | FastAPI routes + SSE — LangChain doesn't replace web frameworks |
| `rate_limit.py` | Cross-cutting infrastructure, not LLM logic |
| `sessions.py` | Storage concern; LangChain has `RunnableWithMessageHistory` but it's overkill for a 30-line dict-based store |
| `evals.py` | You wrote your own eval harness; swapping for LangSmith would be a separate module |
| `static/`, `nginx/`, `systemd/` | Frontend & deployment |

The boundary between "LangChain replaces this" and "LangChain doesn't" is the lesson — most production AI code is *not* the LLM call.

## Project: NovaCRM Support Agent (LangGraph edition)

You'll take the running capstone in `14-framework-migration/` (a verbatim copy of `13-capstone/` minus the Chroma DB) and migrate it to LangChain + LangGraph in four stages, keeping the API contract and tests passing after each stage. The reference baseline lives at `13-capstone/` — do **not** modify it; it's your before-picture for the side-by-side comparison.

End state: `agents.py`, `knowledge.py`, and `semantic_cache.py` rewritten on the framework. `server.py` calls the new code with the same signatures. Same API, same UI, same evals — different internals.

### Requirements
1. After each stage, the FastAPI server still starts and `/chat/sync` returns a valid response for "What plans do you offer?".
2. Langfuse traces still appear (and ideally get richer once the callback handler replaces hand-placed decorators).
3. `evals.py` still passes — same scoring, same dataset.
4. `requirements.txt` updated incrementally, one stage at a time.
5. Final diff vs. `13-capstone/` documented in a `MIGRATION_NOTES.md` you'll write at the end: line counts, dependencies added, observations.

### Starter files
- `agents.py` — **baseline**, identical to `13-capstone/agents.py`. You'll rewrite this in Stages 1 and 3.
- `knowledge.py` — **baseline**. Rewritten in Stage 2.
- `semantic_cache.py` — **baseline**. Lightly touched in Stage 2 (uses `embed_texts` from `knowledge.py`).
- `config.py` — **baseline**. Add LangChain model + embeddings clients here in Stage 1.
- `server.py` — **do not modify**. Imports stay the same name; signatures stay the same.
- `rate_limit.py`, `sessions.py`, `evals.py` — **do not modify**.
- `requirements.txt` — **baseline**. Add deps in this order: `langchain-core`, `langchain-openai` (Stage 1) → `langchain-chroma` (Stage 2) → `langgraph` (Stage 3).

### Your task

#### Stage 1 — Prompts (`ChatPromptTemplate` + `ChatOpenAI`)
1. Add `langchain-core` and `langchain-openai` to `requirements.txt`.
2. In `config.py`, add a `chat_model = ChatOpenAI(model=CHAT_MODEL)` alongside the existing `openai_client`.
3. Convert the four specialist system prompts (`BILLING_PROMPT`, `TECHNICAL_PROMPT`, `GENERAL_PROMPT`, `ESCALATION_PROMPT` — each combined with `SHARED_SYSTEM_PREFIX`) into `ChatPromptTemplate` objects.
4. Convert the router prompt (`ROUTER_PROMPT`) into a `ChatPromptTemplate` and use `chat_model.with_structured_output(RouteDecision)` instead of `openai_client.beta.chat.completions.parse`.
5. Keep `run_agent_loop` using the raw OpenAI client for now — only the *prompt construction* changes in this stage.

**Stop condition:** `route_query("What plans do you offer?")` returns `RouteDecision(category="general", ...)` via the new template + structured output path.

#### Stage 2 — RAG (`Chroma` vectorstore + retriever)
1. Add `langchain-chroma` to `requirements.txt`.
2. Rewrite `knowledge.py` so `embed_texts`, `ingest_file`, `search`, and `list_documents` go through a `Chroma(...)` vectorstore configured with `OpenAIEmbeddings`.
3. Keep the public function signatures exactly as they are (`search(query, top_k) -> list[dict]` etc.) so `agents.py`, `server.py`, and `semantic_cache.py` don't need to change.
4. The output of `search()` should still return `[{"text", "source", "score", "chunk_index"}, ...]` — translate from LangChain `Document` objects.
5. Re-ingest `sample_docs/` into the new vectorstore.

**Stop condition:** `python knowledge.py search "API rate limit"` returns the same top result as the baseline in `13-capstone/`.

#### Stage 3 — Agent loop (`StateGraph`)
1. Add `langgraph` to `requirements.txt`.
2. Define an `AgentState` TypedDict (at minimum: `messages`, plus `cost`/`tools_called` if you want to surface them).
3. Convert each tool in `TOOL_SCHEMAS` to a LangChain `@tool`-decorated function (or keep the dispatch table — your call; document why in the migration notes).
4. Build a `StateGraph` with `agent` and `tools` nodes plus the conditional edge.
5. Replace `run_agent_loop`'s body with a call to `graph.astream(...)` — translate LangGraph events into the existing `AgentEvent(type="status"|"token"|"done"|"error")` shape so the SSE contract holds.
6. The router (`route_query`) and the dispatch in `run_support_agent` can stay procedural — the loop is what becomes a graph.

**Stop condition:** `/chat/sync` returns a streamed response for a tool-using query (e.g. "What's the API rate limit?") with at least one `search_knowledge_base` call visible in the trace.

#### Stage 4 — Observability (Langfuse callback handler)
1. In `config.py`, replace the hand-rolled `observe` import with `from langfuse.langchain import CallbackHandler`.
2. Remove `@observe(...)` decorators from the migrated functions in `agents.py` and `knowledge.py`. Pass `callbacks=[CallbackHandler()]` in the LangGraph + LangChain `config` instead.
3. Verify in Langfuse that retrievals, model calls, and tool executions all trace under one parent span per request — and that span depth/structure is at least as informative as the decorator version.
4. Write `MIGRATION_NOTES.md`: dependencies added, line-count delta vs. `13-capstone/`, two things you liked, two things that felt heavier than the from-scratch version.

**Stop condition:** A single `/chat/sync` call produces one Langfuse trace with nested spans for routing → retrieval → tool calls → final completion, with no `@observe` decorators in your code.

### Hints

<details>
<summary>Stage 1 — combining a shared prefix with specialist prompts</summary>

`ChatPromptTemplate.from_messages` can take a `("system", "...")` tuple where the string is the already-concatenated `SHARED_SYSTEM_PREFIX + BILLING_PROMPT`. You don't need fancy partial composition — just build four templates, one per specialist. The router template is its own thing.
</details>

<details>
<summary>Stage 1 — structured output</summary>

`ChatOpenAI(model=...).with_structured_output(RouteDecision)` returns a runnable that, when invoked, returns a `RouteDecision` directly. No `response.choices[0].message.parsed`. Combine with the prompt: `(router_prompt | chat_model.with_structured_output(RouteDecision)).invoke({"message": message})`.
</details>

<details>
<summary>Stage 2 — keeping the same Chroma data</summary>

Point `Chroma(persist_directory=...)` at the same `.chroma_db` location and use the same `collection_name="knowledge_base"`. The on-disk format is compatible — but the embedding function must match (`text-embedding-3-small`, dimensions=512). If in doubt, `rm -rf .chroma_db` and re-ingest.
</details>

<details>
<summary>Stage 2 — translating Documents back to dicts</summary>

`vectorstore.similarity_search_with_score(query, k=top_k)` returns `[(Document, score), ...]`. Each `Document` has `.page_content` and `.metadata`. Map these back into `{"text": doc.page_content, "source": doc.metadata["source"], "score": score, "chunk_index": doc.metadata["chunk_index"]}`.
</details>

<details>
<summary>Stage 3 — translating LangGraph events to AgentEvent</summary>

`graph.astream(state, stream_mode="updates")` yields dicts keyed by node name with the state delta from that node. `stream_mode="messages"` yields token-level model output. You can pass a list `[stream_mode="updates", "messages"]` to get both — the yielded tuples include a tag telling you which mode produced each event. Use `updates` for `status`/`done` events; use `messages` for `token` events.
</details>

<details>
<summary>Stage 3 — keeping `analyze_image` working</summary>

LangChain's tool decorator can wrap a function with multimodal inputs, but the simplest path is to keep `_tool_analyze_image` as-is and register it as a tool with the same JSON schema. The vision call inside it can stay on the raw OpenAI client — that's a one-shot completion, not part of the agent loop.
</details>

<details>
<summary>Stage 4 — verifying nested spans</summary>

In Langfuse, one HTTP request should produce a single trace with spans like: `route_query` → `search_knowledge_base` (retrieval span) → `agent` (chat completion) → `tools.search_knowledge_base` → `agent` (final completion). If you see flat top-level spans instead of nested, you forgot to pass the same callback handler config through both LangChain and LangGraph calls.
</details>

## Stretch goals
- Replace the in-memory `semantic_cache.py` with `langchain.cache.InMemoryCache` or `RedisSemanticCache` — keep the same hit-rate semantics.
- Wrap `SPECIALISTS` selection inside the graph: add a `router` node that writes `category` to state and a conditional edge to four specialist agent nodes. (This is overkill but instructive.)
- Add a guardrail node before the final response that checks for PII leakage of mock customer data.
- Write a `compare.py` that runs the same 10-question eval set against both `13-capstone/` and `14-framework-migration/` and prints a side-by-side latency / cost / accuracy table.

## Key questions
- Where did LangChain remove duplication, and where did it just rename what you had?
- LangGraph turns your loop into a graph — when does that pay off, and when is a `for` loop fine?
- The retriever interface is uniform across vector stores. What does that uniformity *cost* — what control did you give up?
- The Langfuse callback handler instruments without decorators. What's the trade-off vs. the explicit `@observe` calls in `13-capstone/`?
- After this migration, would you start a brand-new project with LangChain/LangGraph, or build the loop yourself first? Defend your answer in `MIGRATION_NOTES.md`.

## Resources
- [LangChain docs — Get started](https://python.langchain.com/docs/introduction/)
- [LangGraph docs — Concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangGraph — Build a basic chatbot (StateGraph tutorial)](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangChain `Chroma` integration](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [Langfuse — LangChain integration](https://langfuse.com/integrations/frameworks/langchain)
- [LCEL — LangChain Expression Language](https://python.langchain.com/docs/concepts/lcel/)

---

Run `/baby-step` to start Stage 1.
