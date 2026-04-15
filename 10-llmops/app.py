"""
Instrumented LLM API with Langfuse Observability

Usage:
    uvicorn app:app --reload

Endpoints:
    POST /chat       — conversational LLM endpoint
    POST /summarize  — text summarization endpoint
    GET  /metrics    — operational metrics (tokens, cost, latency)
"""

import time
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
import openai
from langfuse import Langfuse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

app = FastAPI(title="Instrumented LLM API")
client = AsyncOpenAI()
langfuse = Langfuse()

# ---------------------------------------------------------------------------
# Model pricing (per token)
# ---------------------------------------------------------------------------

# Prices as of early 2025 — update if models change
MODEL_PRICES = {
    "gpt-4o":       {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":  {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4.1":      {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call in dollars."""
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    return input_tokens * prices["input"] + output_tokens * prices["output"]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"


class SummarizeRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

# TODO: implement MetricsCollector
# - Track: total_requests, total_errors, total_tokens, total_cost, latencies (list)
# - Methods: record_request(), record_error(), p50(), p99(), to_dict()
# - p50/p99: sort the latencies list and pick the value at the percentile index
@dataclass
class MetricsCollector:
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def record_request(self, tokens: int, cost: float, latency: float):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.latencies.append(latency)

    def record_error(self):
        self.total_errors += 1

    def p50(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]

    def p99(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "p50_latency": round(self.p50(), 3),
            "p99_latency": round(self.p99(), 3),
        }


metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# LLM call with Langfuse tracing
# ---------------------------------------------------------------------------

# - Takes: trace (Langfuse trace), span_name, model, messages list
# - Creates a Langfuse generation span on the trace
# - Calls OpenAI chat completions
# - Ends the generation span with output, token usage, and cost
# - Returns a dict: {"content": str, "input_tokens": int, "output_tokens": int, "cost": float}
# - On error: end the span with error status, re-raise the exception
async def track_llm_call(
    trace,
    span_name: str,
    model: str,
    messages: list[dict],
) -> dict:
    # Start a Langfuse generation span (special span type for LLM calls)
    generation = trace.generation(
        name=span_name,
        model=model,
        input=messages,
    )

    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except openai.APIError as exc:
        latency = time.time() - start
        generation.end(
            output=None,
            level="ERROR",
            status_message=str(exc),
            metadata={"latency": latency}
        )
        metrics.record_error()
        raise

    latency = time.time() - start

    usage = response.usage
    cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    content = response.choices[0].message.content

    generation.end(
        output=content,
        usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
        metadata={"cost": cost, "latency": latency},
    )

    return {
        "content": content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost": cost,
        "latency": latency,
    }


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# TODO: implement timing middleware
# - Measure total request duration
# - Add X-Request-Time header to every response
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Request-Time"] = f"{duration:.3f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# - Create a Langfuse trace for the request
# - Create a validation span (just marks that input was validated)
# - Call track_llm_call() with a system prompt and the user's message
# - Update metrics
# - Score the trace based on latency (fast < 2s, normal 2-5s, slow > 5s)
# - Return the response with X-Trace-Id header
@app.post("/chat")
async def chat(request: ChatRequest):
    trace = langfuse.trace(
        name="chat",
        input={"message": request.message, "model": request.model},
    )

    validation = trace.span(name="input-validation")
    validation.end()

    system_prompt = (
        "You are a helpful, concise assistant. "
        "Answer the user's question directly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.message},
    ]

    result = await track_llm_call(
        trace=trace,
        span_name="llm-call",
        model=request.model,
        messages=messages,
    )

    total_tokens = result["input_tokens"] + result["output_tokens"]
    metrics.record_request(
        tokens=total_tokens,
        cost=result["cost"],
        latency=result["latency"],
    )

    latency = result["latency"]
    if latency < 2:
        latency_bucket = "fast"
    elif latency < 5:
        latency_bucket = "normal"
    else:
        latency_bucket = "slow"

    trace.score(name="latency", value=latency_bucket)

    trace.update(output={"response": result["content"]})

    return JSONResponse(
        content={"response": result["content"]},
        headers={"X-Trace-Id": trace.id},
    )

# - Same tracing pattern as /chat but with a summarization system prompt
# - The system prompt should instruct the model to summarize concisely
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    trace = langfuse.trace(
        name="summarize",
        input={"text": request.text, "model": request.model},
    )

    validation = trace.span(name="input-validation")
    validation.end()

    system_prompt = (
        "You are a summarization assistant. "
        "Given a piece of text, produce a concise summary "
        "that captures the key points in 2-3 sentences."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]

    result = await track_llm_call(
        trace=trace,
        span_name="summarize-llm-call",
        model=request.model,
        messages=messages,
    )

    total_tokens = result["input_tokens"] + result["output_tokens"]
    metrics.record_request(
        tokens=total_tokens,
        cost=result["cost"],
        latency=result["latency"],
    )

    latency = result["latency"]
    if latency < 2:
        latency_bucket = "fast"
    elif latency < 5:
        latency_bucket = "normal"
    else:
        latency_bucket = "slow"

    trace.score(name="latency", value=latency_bucket)

    trace.update(output={"summary": result["content"]})

    return JSONResponse(
        content={"summary": result["content"]},
        headers={"X-Trace-Id": trace.id},
    )


# TODO: implement GET /metrics
# - Return the metrics collector as JSON
@app.get("/metrics")
async def get_metrics():
    return metrics.to_dict()
