# Module 07 — Streaming & Production Patterns

**Goal:** Build a production-ready FastAPI service that streams LLM responses token-by-token, handles errors gracefully, caches repeated queries, and serves concurrent users without falling over.

**Time:** ~2 days

---

## Setup & running

```bash
pip install fastapi uvicorn openai python-dotenv redis httpx sse-starlette

# Start Redis (for caching)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run the API server
uvicorn api:app --reload --port 8000

# Test streaming from the CLI
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG in 3 sentences"}'
```

---

## What you'll learn

- How streaming works under the hood (Server-Sent Events, chunked transfer)
- Why streaming matters for UX (time-to-first-token vs. total latency)
- Async Python with `asyncio` and how FastAPI uses it
- Production error handling: retries, timeouts, circuit breakers
- Caching LLM responses to cut cost and latency
- Rate limiting and concurrent request management
- Health checks and graceful shutdown

---

## Concepts

### Streaming: why it matters

Without streaming, the user stares at a blank screen for 3–10 seconds while the LLM generates the full response. With streaming, the first token appears in ~200ms. The total time is the same, but the perceived experience is dramatically better.

Under the hood, streaming uses **Server-Sent Events (SSE)** — a simple HTTP protocol where the server sends a stream of `data:` lines over a long-lived connection:

```
data: {"token": "The"}
data: {"token": " capital"}
data: {"token": " of"}
data: {"token": " France"}
data: {"token": " is"}
data: {"token": " Paris"}
data: [DONE]
```

The OpenAI SDK makes this easy:

```python
stream = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### Async Python for I/O-bound work

LLM API calls are I/O-bound — your server is waiting on a network response. With synchronous code, each request blocks a thread. With `async/await`, one thread handles many requests by yielding control while waiting:

```python
# Sync: blocks the thread during the API call
def get_response(msg):
    return client.messages.create(...)

# Async: yields the thread while waiting
async def get_response(msg):
    return await async_client.messages.create(...)
```

FastAPI is async-native. Use `async def` for your route handlers and the async version of the OpenAI SDK (`AsyncOpenAI`).

### Caching LLM responses

LLM calls are slow (~1-5s) and expensive (~$0.003-0.06 per call). If the same question comes in twice, serve the cached answer:

```
Request → Hash the input → Check cache → Hit? Return cached → Miss? Call LLM → Store in cache → Return
```

Two cache strategies:
- **Exact match**: hash the full prompt. Simple but only catches identical queries.
- **Semantic cache**: embed the query, find nearest neighbor in cache. Catches paraphrases but adds latency for the embedding call.

Start with exact match (Redis). Add semantic caching as a stretch goal.

### Production error handling

LLM APIs fail. Networks fail. Rate limits hit. Your server must handle all of this:

| Failure | Pattern | Implementation |
|---|---|---|
| Transient API error (500, 529) | **Retry with exponential backoff** | 3 retries, 1s → 2s → 4s delay |
| Rate limit (429) | **Backoff + respect `Retry-After` header** | Parse header, sleep, retry |
| Timeout | **Request timeout + fallback** | 30s timeout, return graceful error |
| Persistent failure | **Circuit breaker** | After N failures, stop calling for M seconds |

The OpenAI SDK has built-in retries. But you should understand the pattern and add your own layer for custom behavior.

### Rate limiting

Protect your server from abuse. Two levels:
- **Per-user**: max 10 requests/minute per API key
- **Global**: max 100 requests/minute total (to stay within your LLM API quota)

Use a simple token bucket or sliding window counter in Redis.

### Health checks and graceful shutdown

Production APIs need:
- `GET /health` — returns 200 if the server is up and dependencies (Redis, LLM API) are reachable
- Graceful shutdown — finish in-flight streaming requests before stopping
- Request IDs — attach a unique ID to every request for tracing

---

## Project: Production Streaming API

Build a FastAPI service that wraps GPT-4o-mini as a streaming chat API, with caching, error handling, rate limiting, and health checks.

### Requirements

```
- FastAPI server with two main endpoints:
  1. POST /chat — non-streaming, returns full response as JSON
  2. POST /chat/stream — streaming via SSE, tokens sent as they arrive
- Both endpoints accept: {"message": "...", "conversation_id": "optional"}
- Conversation memory: track multi-turn conversations by conversation_id
- Redis caching:
  - Cache responses by (message + conversation_id) hash
  - Cache TTL: 1 hour
  - Return X-Cache: HIT/MISS header
- Error handling:
  - Retry transient failures (3 attempts, exponential backoff)
  - 30-second timeout on LLM calls
  - Structured error responses: {"error": "...", "request_id": "..."}
- Rate limiting:
  - 20 requests/minute per client IP
  - Return 429 with Retry-After header when exceeded
- Observability:
  - GET /health — checks Redis and returns server status
  - Request ID on every response (X-Request-ID header)
  - Log request duration, token count, cache hit/miss
- Graceful streaming:
  - If client disconnects mid-stream, stop the LLM call
  - Send a final [DONE] event when streaming completes
```

### Starter code

```python
# api.py
import os
import json
import time
import uuid
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import redis.asyncio as redis
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Config ───────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
CACHE_TTL = 3600  # 1 hour
RATE_LIMIT = 20   # requests per minute per IP
RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT = 30  # seconds

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Lifespan (startup / shutdown) ────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: initialize Redis connection pool and OpenAI async client
    # Store them on app.state so route handlers can access them
    # app.state.redis = ...
    # app.state.openai = ...
    yield
    # TODO: close connections on shutdown


app = FastAPI(title="LLM Streaming API", lifespan=lifespan)


# ── Models ───────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    request_id: str
    cached: bool
    duration_ms: float


# ── Middleware ────────────────────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique request ID to every request/response."""
    # TODO: generate a UUID, attach it to request.state, add X-Request-ID header
    pass


# ── Rate limiting ────────────────────────────────────────────────


async def check_rate_limit(client_ip: str, redis_client) -> bool:
    """
    Sliding window rate limiter using Redis.
    Returns True if the request is allowed, False if rate limited.
    """
    # TODO: implement sliding window counter in Redis
    # Key: f"rate:{client_ip}"
    # Use a sorted set with timestamps, or a simple counter with TTL
    pass


# ── Caching ──────────────────────────────────────────────────────


def make_cache_key(message: str, conversation_id: str) -> str:
    """Create a deterministic cache key from the request."""
    # TODO: hash the message + conversation_id
    pass


async def get_cached(redis_client, cache_key: str) -> str | None:
    """Look up a cached response."""
    # TODO: get from Redis, return None on miss
    pass


async def set_cached(redis_client, cache_key: str, response: str):
    """Store a response in the cache."""
    # TODO: set in Redis with TTL
    pass


# ── LLM calls ───────────────────────────────────────────────────


async def call_llm(client: AsyncOpenAI, messages: list[dict]) -> str:
    """
    Call GPT-4o-mini with retry logic and timeout.
    Retries up to RETRY_ATTEMPTS times on transient errors.
    """
    # TODO: implement retry loop with exponential backoff
    # Use asyncio.wait_for() for timeout
    # Catch openai.APIStatusError for retryable errors (500, 529)
    pass


async def stream_llm(client: AsyncOpenAI, messages: list[dict]):
    """
    Stream GPT-4o-mini's response token by token.
    Yields individual text chunks as they arrive.
    """
    # TODO: use client.chat.completions.create(stream=True)
    # Yield each chunk.choices[0].delta.content as it arrives
    # Handle timeout and errors mid-stream
    pass


# ── Conversation memory ─────────────────────────────────────────


async def get_conversation(redis_client, conversation_id: str) -> list[dict]:
    """Load conversation history from Redis."""
    # TODO: get conversation messages from Redis
    pass


async def save_conversation(redis_client, conversation_id: str, messages: list[dict]):
    """Save updated conversation history to Redis."""
    # TODO: save conversation messages to Redis with TTL
    pass


# ── Routes ───────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Non-streaming chat endpoint. Returns the full response as JSON."""
    # TODO:
    # 1. Check rate limit
    # 2. Check cache
    # 3. Load conversation history
    # 4. Call LLM (with retries and timeout)
    # 5. Save to cache and conversation history
    # 6. Return ChatResponse with timing and metadata
    pass


@app.get("/chat/stream")
async def chat_stream(request: Request):
    """
    Streaming chat endpoint via SSE.
    Sends tokens as server-sent events as they arrive from the LLM.
    """
    # TODO:
    # 1. Parse request body
    # 2. Check rate limit
    # 3. Check cache — if hit, stream the cached response with small delays
    # 4. Load conversation history
    # 5. Stream LLM response, yielding SSE events
    # 6. After stream completes, save to cache and conversation history
    # 7. Send [DONE] event
    # 8. If client disconnects, stop the stream
    pass


@app.get("/health")
async def health():
    """Health check — verify Redis and report server status."""
    # TODO: ping Redis, return status with uptime and version info
    pass


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
```

### Your task

1. **Lifespan** — Initialize Redis + OpenAI clients on startup, close on shutdown
2. **`add_request_id` middleware** — UUID on every request, `X-Request-ID` header on every response
3. **`check_rate_limit()`** — Sliding window counter in Redis, 20 req/min per IP
4. **`make_cache_key()` + `get_cached()` + `set_cached()`** — Hash-based caching with TTL
5. **`call_llm()`** — Async call with retry (exponential backoff) and timeout
6. **`stream_llm()`** — Async generator yielding text deltas from OpenAI's streaming API
7. **`get_conversation()` + `save_conversation()`** — Multi-turn memory via Redis
8. **`POST /chat`** — Full non-streaming endpoint wiring it all together
9. **`POST /chat/stream`** — SSE streaming endpoint with client disconnect detection
10. **`GET /health`** — Health check endpoint

### Hints

<details>
<summary>Lifespan — initializing async clients</summary>

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    app.state.openai = AsyncOpenAI()
    yield
    await app.state.redis.close()
```
</details>

<details>
<summary>Rate limiting — sliding window with sorted set</summary>

```python
async def check_rate_limit(client_ip: str, redis_client) -> bool:
    key = f"rate:{client_ip}"
    now = time.time()
    window_start = now - 60

    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # remove old entries
    pipe.zadd(key, {str(now): now})               # add current request
    pipe.zcard(key)                                # count requests in window
    pipe.expire(key, 60)                           # auto-cleanup
    results = await pipe.execute()

    return results[2] <= RATE_LIMIT
```
</details>

<details>
<summary>Streaming — async generator with SSE</summary>

```python
async def stream_llm(client, messages):
    stream = await client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
```

In the route handler:
```python
async def event_generator():
    full_response = []
    async for chunk in stream_llm(client, messages):
        full_response.append(chunk)
        yield {"event": "token", "data": json.dumps({"token": chunk})}
    yield {"event": "done", "data": json.dumps({"full_response": "".join(full_response)})}

return EventSourceResponse(event_generator())
```
</details>

<details>
<summary>Client disconnect detection</summary>

```python
async def event_generator():
    try:
        async for chunk in stream_llm(client, messages):
            if await request.is_disconnected():
                logger.info("Client disconnected, stopping stream")
                break
            yield {"event": "token", "data": json.dumps({"token": chunk})}
    except asyncio.CancelledError:
        logger.info("Stream cancelled")
```
</details>

<details>
<summary>Retry with exponential backoff</summary>

```python
async def call_llm(client, messages):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return response.choices[0].message.content
        except (asyncio.TimeoutError, openai.APIStatusError) as e:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
```
</details>

---

## Stretch goals

- **Semantic caching** — embed queries with OpenAI, find nearest neighbor in a cache index. Serve cached response if similarity > 0.95.
- **WebSocket endpoint** — add `WS /chat/ws` for bidirectional streaming (lower overhead than SSE for high-frequency updates)
- **Token counting** — track input/output tokens per request, expose `GET /stats` with total usage and estimated cost
- **Model fallback** — if GPT-4o-mini is down, fall back to GPT-4o or another provider automatically
- **Request queuing** — when the LLM API is overloaded, queue requests instead of failing immediately; use Redis as the queue backend
- **Structured output streaming** — stream JSON responses token by token, parsing partial JSON as it arrives
- **Load testing** — use `locust` or `k6` to stress-test your API; find the concurrency limit

---

## Key questions to answer before moving on

1. What's the difference between `async def` and `def` in a FastAPI route handler? When does each block?
2. Why use SSE instead of WebSockets for LLM streaming? When would you prefer WebSockets?
3. How do you prevent cache poisoning — a bad response getting cached and served to many users?
4. What happens to in-flight streaming requests when the server shuts down? How do you handle this gracefully?
5. How would you implement semantic caching without adding too much latency to cache lookups?

---

## Resources

- [FastAPI — Streaming Responses](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [OpenAI SDK — Streaming](https://platform.openai.com/docs/guides/streaming)
- [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Redis — Rate Limiting patterns](https://redis.io/docs/latest/develop/use/patterns/)
- [Resilience patterns — Circuit Breaker, Retry, Timeout](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)

---

**When done:** Mark Module 07 as shipped in the root README, commit, and move to [Module 08](../08-fine-tuning/).
