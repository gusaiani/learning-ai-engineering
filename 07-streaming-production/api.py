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
import openai
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
    # Startup: create shared clients
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    app.state.openai = AsyncOpenAI()
    yield
    # Shutdown: clean up
    await app.state.redis.close()

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
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Rate limiting ────────────────────────────────────────────────


async def check_rate_limit(client_ip: str, redis_client) -> bool:
    """
    Sliding window rate limiter using Redis.
    Returns True if the request is allowed, False if rate limited.
    """
    key = f"rate:{client_ip}"
    now = time.time()
    window_start = now - 60

    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # drop old entries
    pipe.zadd(key, {str(now): now})              # add this request
    pipe.zcard(key)                              # count in window
    pipe.expire(key, 60)                         # auto-cleanup
    results = await pipe.execute()

    return results[2] <= RATE_LIMIT


# ── Caching ──────────────────────────────────────────────────────


def make_cache_key(message: str, conversation_id: str) -> str:
    """Create a deterministic cache key from the request."""
    raw = f"{conversation_id}: {message}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"cache:{digest}"


async def get_cached(redis_client, cache_key: str) -> str | None:
    """Look up a cached response."""
    return await redis_client.get(cache_key)

async def set_cached(redis_client, cache_key: str, response: str):
    """Store a response in the cache."""
    await redis_client.set(cache_key, response, ex=CACHE_TTL)

# ── LLM calls ───────────────────────────────────────────────────


async def call_llm(client: AsyncOpenAI, messages: list[dict]) -> str:
    """
    Call GPT-4o-mini with retry logic and timeout.
    Retries up to RETRY_ATTEMPTS times on transient errors.
    """
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
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
            await asyncio.sleep(wait)

async def stream_llm(client: AsyncOpenAI, messages: list[dict]):
    """
    Stream GPT-4o-mini's response token by token.
    Yields individual text chunks as they arrive.
    """
    stream = await asyncio.wait_for(
        client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            stream=True,
        ),
        timeout=REQUEST_TIMEOUT,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content 


# ── Conversation memory ─────────────────────────────────────────


async def get_conversation(redis_client, conversation_id: str) -> list[dict]:
    """Load conversation history from Redis."""
    data = await redis_client.get(f"conv:{conversation_id}")
    if data is None:
        return []
    return json.loads(data)


async def save_conversation(redis_client, conversation_id: str, messages: list[dict]):
    """Save updated conversation history to Redis."""
    await redis_client.set(
        f"conv:{conversation_id}",
        json.dumps(messages),
        ex=CACHE_TTL
    )


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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    Streaming chat endpoint via SSE.
    Sends tokens as server-sent events as they arrive from the LLM.
    """
    # TODO:
    # 1. Check rate limit
    # 2. Check cache — if hit, stream the cached response with small delays
    # 3. Load conversation history
    # 4. Stream LLM response, yielding SSE events
    # 5. After stream completes, save to cache and conversation history
    # 6. Send [DONE] event
    # 7. If client disconnects, stop the stream
    pass


@app.get("/health")
async def health():
    """Health check — verify Redis and report server status."""
    try:
        await app.state.redis.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "redis": str(e)},
        )


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
