# Module 10 — LLMOps & Observability

**Goal:** Take a working LLM-powered API and instrument it so you can see every call, trace every request, track costs, and catch problems before users do.

**Time:** ~2 days

---

## Setup & running

```bash
pip install openai python-dotenv fastapi uvicorn langfuse

# Run the instrumented API
uvicorn app:app --reload

# Test endpoints
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG in one paragraph"}'

curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" \
  -d '{"text": "Long article text here..."}'

curl http://localhost:8000/metrics
```

You'll need a Langfuse account (free tier works). Add to your `.env`:

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## What you'll learn

- Why observability matters more for LLM apps than traditional software
- Tracing: tracking a request across multiple LLM calls
- Spans: breaking a trace into logical steps (retrieval, generation, post-processing)
- Cost tracking: knowing exactly how much each request costs in real-time
- Latency monitoring: measuring time-to-first-token and total response time
- Error classification: distinguishing API failures from bad outputs
- Building a `/metrics` endpoint that exposes operational health

---

## Concepts

### Why LLM observability is different

Traditional software is deterministic — same input, same output. LLM apps are stochastic. A request that works 99% of the time can silently degrade: the model hallucinates, the response is off-topic, latency spikes 3x because of a longer prompt. You can't unit-test your way out of this.

Observability for LLM apps means tracking three things you don't track in normal software:

1. **What the model actually said** — not just "200 OK", but the full prompt/response pair
2. **How much it cost** — every token has a price, and a bug in your prompt template can 10x your bill overnight
3. **How long it took** — LLM latency varies wildly based on prompt length, model load, and output length

### Traces and spans

A **trace** represents one end-to-end user request. A **span** is a step within that trace.

```
Trace: POST /chat
├── Span: input-validation (2ms)
├── Span: llm-call (1,240ms)
│   ├── model: gpt-4o-mini
│   ├── input_tokens: 342
│   ├── output_tokens: 187
│   └── cost: $0.0003
└── Span: response-formatting (1ms)
Total: 1,243ms | Cost: $0.0003
```

For a RAG endpoint, you'd see more spans:

```
Trace: POST /ask
├── Span: embed-query (85ms)
├── Span: vector-search (23ms)
├── Span: build-prompt (1ms)
├── Span: llm-generation (2,100ms)
└── Span: format-response (2ms)
```

This breakdown lets you answer: "Why is this endpoint slow?" Is it the embedding call? The database? The generation? Without spans, you just know it's slow.

### Langfuse

Langfuse is an open-source LLM observability platform. You can self-host it or use their cloud (free tier: 50k observations/month).

Key concepts:

| Concept | What it is | When to use |
|---------|-----------|-------------|
| **Trace** | One end-to-end request | Every API call creates one |
| **Span** | A step within a trace | Wrap each logical step |
| **Generation** | A special span for LLM calls | Automatically captures model, tokens, cost |
| **Score** | A quality metric attached to a trace | User feedback, LLM-as-judge, latency thresholds |

Basic usage with the Langfuse Python SDK:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create a trace for the request
trace = langfuse.trace(name="chat", input={"message": user_message})

# Create a span for the LLM call
generation = trace.generation(
    name="llm-call",
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
)

# After the LLM responds
generation.end(
    output=response_text,
    usage={"input": prompt_tokens, "output": completion_tokens},
)

trace.update(output={"response": response_text})
```

### Cost tracking

OpenAI pricing changes, but the pattern is always: `cost = input_tokens * input_price + output_tokens * output_price`. You need a price table:

```python
MODEL_PRICES = {
    "gpt-4o":      {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4.1":     {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},
    "gpt-4.1-mini":{"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
}
```

Track cumulative cost per request and per endpoint. A sudden cost spike usually means a prompt is being constructed wrong (e.g., including the entire document instead of a chunk).

### Latency buckets

Not all latency is equal. Track these separately:

- **Time to first token (TTFT):** How long before the user sees anything. Critical for streaming UIs.
- **Total response time:** End-to-end. Matters for API consumers.
- **LLM call time:** Just the model inference. If this is slow, it's OpenAI's problem. If total is slow but this is fast, it's your code.

### In-memory metrics

For a production API, you want a `/metrics` endpoint that returns current operational stats without needing to query Langfuse. Keep a simple in-memory counter:

```python
@dataclass
class Metrics:
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def p50(self) -> float:
        ...
    def p99(self) -> float:
        ...
```

This gives you instant health checks. Langfuse gives you deep historical analysis. Use both.

### Error classification

LLM errors come in flavors:

| Type | Example | How to detect | Action |
|------|---------|--------------|--------|
| **API error** | Rate limit, timeout, 500 | HTTP status code | Retry with backoff |
| **Bad output** | Hallucination, wrong format | Validation / LLM-as-judge | Log, alert, maybe retry |
| **Slow response** | >5s for a simple query | Latency threshold | Log, investigate |
| **Cost anomaly** | 10x normal token count | Cost threshold | Alert, investigate prompt |

In your instrumented app, catch and classify all four types. Log them differently so you can filter in Langfuse.

---

## Project: Instrumented LLM API

Build a FastAPI app with two LLM-powered endpoints, fully instrumented with Langfuse tracing, cost tracking, latency monitoring, and a metrics dashboard endpoint.

### Architecture

```
Client request
    │
    ▼
FastAPI endpoint
    │
    ├── Langfuse trace created
    │
    ├── Span: input validation
    │
    ├── Span: LLM call (Generation)
    │   ├── captures model, tokens, cost
    │   └── captures latency
    │
    ├── Span: response formatting
    │
    ├── Metrics updated (in-memory)
    │
    └── Trace closed
    │
    ▼
Response + trace ID in headers
```

### Requirements

1. **Two LLM endpoints:**
   - `POST /chat` — conversational endpoint (system prompt + user message)
   - `POST /summarize` — takes a text body, returns a summary
2. **Langfuse tracing** on every request — each endpoint creates a trace with proper spans
3. **Generation spans** for LLM calls — capture model, input/output tokens, and cost
4. **Cost calculation** — compute cost from token counts using a price table
5. **Latency tracking** — measure and record total request time and LLM call time
6. **Error handling** — catch API errors, classify them, log to Langfuse with error spans
7. **`GET /metrics` endpoint** — returns JSON with: total requests, total errors, total cost, total tokens, p50/p99 latency
8. **Trace ID in response headers** — every response includes `X-Trace-Id` so you can look it up in Langfuse
9. **Request middleware** that times every request (not just LLM endpoints)
10. **Score logging** — log a latency score on each trace (e.g., "fast" if <2s, "slow" if >5s)

### Starter files

- `app.py` — FastAPI app with all endpoints and instrumentation stubbed out

### Your task

1. Set up the Langfuse client and model price table
2. Implement the `track_llm_call()` helper — wraps an OpenAI call with Langfuse generation tracking
3. Implement `POST /chat` with full tracing
4. Implement `POST /summarize` with full tracing
5. Build the in-memory `MetricsCollector` class
6. Implement `GET /metrics` with p50/p99 latency
7. Add the timing middleware
8. Add error handling and classification
9. Add trace ID to response headers
10. Test all endpoints and verify traces appear in Langfuse

### Hints

<details>
<summary>Hint for step 2</summary>
The `track_llm_call` function should create a Langfuse generation span, make the OpenAI call, then end the span with token usage and cost. Return both the LLM response and the cost so the caller can update metrics.
</details>

<details>
<summary>Hint for step 5</summary>
Use a list to store latencies and compute percentiles with `sorted(latencies)[int(len(latencies) * percentile)]`. Thread safety isn't critical for a learning project, but in production you'd use a lock or a thread-safe data structure.
</details>

<details>
<summary>Hint for step 7</summary>
FastAPI middleware uses `@app.middleware("http")`. Call `await call_next(request)` and measure the time before/after. Add the duration as a response header too.
</details>

<details>
<summary>Hint for step 8</summary>
Wrap the OpenAI call in a try/except. Catch `openai.APIError` for API-level errors. For output quality issues, you'd use evals — but for this module, just catch structural errors (e.g., empty response, JSON parse failure).
</details>

---

## Stretch goals

- **Streaming with traces** — stream the `/chat` response while still capturing full token counts and latency in Langfuse
- **Langfuse dashboard** — set up custom dashboards in the Langfuse UI showing cost per endpoint, latency trends, and error rates
- **Alerting** — add a simple threshold check that prints a warning when cost per request exceeds $0.01 or latency exceeds 10s

---

## Key questions

- Why can't you rely solely on HTTP status codes to know if an LLM endpoint is working correctly?
- What's the difference between monitoring (is it up?) and observability (why is it broken?)?
- How would you detect a prompt regression — a change that makes outputs worse without causing errors?
- If your `/metrics` endpoint shows p99 latency at 8s but p50 at 1.2s, what does that tell you?
- What's the risk of logging full prompts and responses to an observability platform?

---

## Resources

- [Langfuse Python SDK docs](https://langfuse.com/docs/sdk/python)
- [OpenAI usage/pricing docs](https://platform.openai.com/docs/api-reference/chat)
- [FastAPI middleware docs](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Google SRE Book — Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
