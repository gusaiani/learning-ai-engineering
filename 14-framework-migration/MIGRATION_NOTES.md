  # Migration Notes — From-Scratch → LangChain + LangGraph

  ## Scope

  Three files moved onto the framework: `agents.py` (prompts, router, agent loop), `knowledge.py` (RAG ingestion + search), and `config.py` (added a LangChain `ChatOpenAI`, a Langfuse `CallbackHandler`, and kept the existing `OpenAI` client for the vision call inside `analyze_image`). Everything else — `server.py` (FastAPI + SSE), `rate_limit.py`, `sessions.py`, `evals.py`, `semantic_cache.py`, the static frontend, nginx config, and systemd units — was left untouched. Not because the migration ran out of runway, but because LangChain doesn't compete with web frameworks, rate limiters, dict-based session stores, hand-rolled eval harnesses, or anything outside the LLM call. The boundary between "this gets a framework" and "this stays as plain Python" turned out to be the most important deliverable of the migration: most production AI code is *not* the LLM call, and a framework that tries to own those edges (e.g. `RunnableWithMessageHistory` for sessions) would have added weight for no benefit at this scale. The API contract held: same routes, same SSE event shape, same `/chat/sync` and `/chat/stream` endpoints, same evals, same UI.

  ## Dependencies added

  Four new top-level packages, no removals:

  | Package | Version | Used for |
  |---|---|---|
  | `langchain-core` | 1.3.2 | `ChatPromptTemplate`, `@tool`, message types (`SystemMessage`, `AIMessage`, `ToolMessage`) |
  | `langchain-openai` | 1.2.1 | `ChatOpenAI`, `OpenAIEmbeddings` |
  | `langchain-chroma` | 1.1.0 | `Chroma` vectorstore wrapper around the existing `chromadb` client |
  | `langgraph` | 1.0.4 | `StateGraph`, `add_messages` reducer, conditional edges |

  The raw `openai` client stayed (used inside `analyze_image` for the vision call — a one-shot completion that doesn't sit on the agent loop, so wrapping it would buy nothing). `chromadb` stayed too: `langchain-chroma` is a thin adapter that takes an existing `chromadb.PersistentClient`, not a replacement. Transitive footprint is real — installing the four adds ~150 MB and pulls in `pydantic-core`, `tiktoken`, `dataclasses-json`, `tenacity`, `jsonpatch`, `orjson`, and the LangSmith client even though we don't use LangSmith. That cost is invisible in `requirements.txt` but shows up in container size and cold-start time.

  ## Line-count delta

  | File | 13-capstone | 14-migration | Δ |
  |---|---:|---:|---:|
  | `agents.py` | 635 | 609 | −26 |
  | `knowledge.py` | 205 | 200 | −5 |
  | `config.py` | 70 | 80 | +10 |
  | **Total** | **910** | **889** | **−21** |

  Net change is roughly noise — 21 lines on 910 is 2.3%. The interesting story is *where bytes moved*, not the totals.

  **Shrank dramatically:**
  - `TOOL_SCHEMAS`: 94 lines of hand-written JSON Schema → gone. `@tool` derives the schema from the function signature + docstring.
  - `execute_tool` + 5 `_tool_*` dispatch helpers: ~75 lines → replaced by `tools_by_name = {t.name: t for t in TOOLS}` and a 12-line `run_tools` node.
  - Router body: `openai_client.beta.chat.completions.parse(...).choices[0].message.parsed` → `(prompt | model.with_structured_output(RouteDecision)).invoke(...)`. About 8 lines down to 2.
  - Streaming usage extraction: hand-walking `chunk.usage.prompt_tokens_details.cached_tokens` → `chunk.usage_metadata["input_token_details"]["cache_read"]`. Cross-provider keys, normalized.
  - `@observe` decorators on 4 functions → one `start_as_current_span` context manager wrapping the whole request.

  **Grew:**
  - `AgentState` TypedDict + `add_messages` annotation + three node functions (`call_model`, `should_continue`, `run_tools`) + `StateGraph` builder + edge mapping: ~50 lines that didn't exist before, expressing what was a 50-line `for _ in range(10):` loop.
  - LangGraph stream → `AgentEvent` translation: ~30 lines of marshalling. Previously the loop produced `AgentEvent` directly because we owned the stream loop; now the framework streams its own event shape and we translate.
  - Three import blocks: `langchain_core.{prompts, tools, messages}`, `langgraph.{graph, graph.message}`, `langfuse.langchain.CallbackHandler`. Each adds 1–2 lines but the cognitive surface is much larger than a single line implies.

  ## What I liked

  **1. Tool definition collapsed to one source of truth.** Before, every tool existed in three places: a `_tool_*` handler, a `TOOL_SCHEMAS` JSON Schema entry, and a dispatch arm in `execute_tool`. Adding a tool meant editing three blocks and keeping them in sync. With `@tool`, the function *is* the tool — schema derived from the type hints, dispatch derived from `tools_by_name = {t.name: t for t in TOOLS}`. Adding a new tool is now a single decorated function. That's the kind of duplication-removal that justifies a framework.

  **2. The graph shape became inspectable data.** `graph.get_graph().draw_mermaid()` produces a diagram. Per-node testing is trivial — `call_model({"messages": [...]})` returns a state patch you can assert on without running the whole loop. Adding a guardrail node before the final response is one `add_node` + one redirected edge, not a refactor of the loop body. The from-scratch `for _ in range(10):` was tight and readable, but it was *opaque* — you couldn't query "what nodes does this agent visit?" without reading the code.

  **3. Cross-provider message and usage normalization.** `chunk.usage_metadata` is the same shape whether the underlying call is OpenAI, Anthropic, or Bedrock. The from-scratch code was hardcoded to `chunk.usage.prompt_tokens_details.cached_tokens` — switching providers would have meant rewriting the usage extraction. Same story for `AIMessage.tool_calls`: a normalized list of `{"name", "args", "id"}` dicts regardless of which provider produced them. This isn't visible in line count but it's where the framework genuinely earns its keep on a multi-provider codebase.

  **4. Observability without per-function decoration.** One `langfuse_handler = CallbackHandler()` in `config.py`, one `start_as_current_span` wrapping `run_support_agent`, and every model call, retrieval, and tool execution traces under one parent span automatically. Adding a new node = automatic span. Before, each new function meant remembering to add `@observe(name="...")` and getting the nesting right.

  ## What felt heavier than from-scratch

  **1. The graph stream → SSE translation is mandatory boilerplate.** LangGraph streams `(stream_mode, event)` tuples in two flavors (`"updates"` and `"messages"`). Our SSE contract wants `AgentEvent(type=...)`. The 30 lines of "if `stream_mode == "messages"`, check `metadata.langgraph_node`, filter empty `chunk.content`, accumulate `usage_metadata`" is pure marshalling — nothing about it is interesting business logic, and you have to write it. The from-scratch loop produced `AgentEvent` directly because we owned the loop; the framework owns the loop now and we translate at the seam. This is the framework tax for using `graph.stream`. If you only need `graph.invoke`, this disappears — but then you lose token streaming, which is non-negotiable for a chat UI.

  **2. Per-specialist tool subsets quietly disappeared.** In `13-capstone/`, the `general` specialist had only `search_knowledge_base`, `escalation` had only `create_ticket`, billing/technical had everything. That was enforced by passing different `tools` arrays into `run_agent_loop`. With `chat_model.bind_tools(TOOLS)` bound globally to one graph, every specialist now has access to every tool. To preserve the old behavior you'd compile four graphs (one per specialist) or pass the tool list through state and rebind dynamically — both options are heavier than the from-scratch dict literal `{"general": {"tools": [TOOL_SCHEMAS[0]]}}`. The framework has an answer (subgraphs, conditional edges into specialist nodes) but it's the README's stretch goal for a reason: it's overkill for what was three lines of config.

  **3. Type signatures became fuzzier.** `AIMessage.tool_calls` returns `list[dict]` but the dict is loosely typed — `tool_call["name"]`, `tool_call["args"]`, `tool_call["id"]`, with no IDE autocomplete and no compile-time check that you spelled the keys right. `state["messages"]` is `Annotated[list, add_messages]` — what's actually in the list at any point depends on which node ran last. The from-scratch code was concrete: you knew `full_messages` was `list[dict]` with a fixed `{"role", "content"}` shape, and the OpenAI SDK had typed responses. Now half the values are LangChain `BaseMessage` subclasses and half are dicts (when seeded from `run_support_agent`), and the `add_messages` reducer handles the conversion implicitly. It works, but you trade static legibility for dynamic flexibility.

  **4. Transitive dependency surface.** Four explicit deps pulled in ~20 transitive packages including LangSmith client code we don't use. None of this shows up in `requirements.txt` until something breaks (a `pydantic` major bump, a `tenacity` change) and you discover you're now downstream of dependency politics you didn't sign up for. The from-scratch version had `openai`, `chromadb`, `langfuse` — a dependency graph small enough to audit in an afternoon. The migrated version is not.

  ## Would I start a new project on this stack?

  **Confidence: high. Answer: it depends — and the dependencies are specific.**

  *Yes, start with LangGraph from day one if:*
  - The agent loop has more than two branches (router → specialists → tools → guardrails → output formatter, etc.). At that point a `for` loop with nested conditionals becomes harder to read than a graph.
  - You need to support more than one model provider (OpenAI + Anthropic, or OpenAI + a local model). The message and usage normalization pays back immediately.
  - You expect to swap vector stores (prototyping on Chroma, planning on Pinecone). The retriever abstraction is genuinely useful here — no other refactor in this migration was a one-import change.
  - You need automatic tracing under one parent span and don't want to litter `@observe` everywhere.

  *No, build the loop yourself first if:*
  - You're on a single provider, single vector store, and the agent loop is "model → tools → model → done." A 50-line `for _ in range(10):` is faster to read, faster to debug, has no transitive dependencies, and doesn't require knowing what `add_messages` is. The from-scratch capstone in `13-capstone/` is *more* legible to a new contributor than the migrated version, full stop. Frameworks are leverage; leverage is also weight, and weight is only worth carrying when there's something to lift.
  - You want clean static types end-to-end. The framework's runtime polymorphism is convenient but it costs you compile-time guarantees.
  - The team doesn't already know LangChain. The learning curve is real (LCEL, runnable config, callback propagation, stream modes, message reducers) and reading the from-scratch code is a 30-minute exercise.

  The honest pattern I'd follow now: build from-scratch first to prove the loop works and to internalize the model. Migrate to LangGraph the day a second branch, a second provider, or a second vector store enters scope — not before. Doing this migration *as a learning exercise on a working codebase* turned out to be the right way to learn what the framework does, because every abstraction had a from-scratch counterpart to compare against. Reading the LangChain docs cold would have left me with vibes-level understanding of `bind_tools` and `with_structured_output`; rewriting working code into them made the trade-offs concrete.
