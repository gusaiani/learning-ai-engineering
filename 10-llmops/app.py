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
        pass

    def record_error(self):
        pass

    def p50(self) -> float:
        pass

    def p99(self) -> float:
        pass

    def to_dict(self) -> dict:
        pass


metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# LLM call with Langfuse tracing
# ---------------------------------------------------------------------------

# TODO: implement track_llm_call()
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
    pass


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# TODO: implement timing middleware
# - Measure total request duration
# - Add X-Request-Time header to every response
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# TODO: implement POST /chat
# - Create a Langfuse trace for the request
# - Create a validation span (just marks that input was validated)
# - Call track_llm_call() with a system prompt and the user's message
# - Update metrics
# - Score the trace based on latency (fast < 2s, normal 2-5s, slow > 5s)
# - Return the response with X-Trace-Id header
@app.post("/chat")
async def chat(request: ChatRequest):
    pass


# TODO: implement POST /summarize
# - Same tracing pattern as /chat but with a summarization system prompt
# - The system prompt should instruct the model to summarize concisely
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    pass


# TODO: implement GET /metrics
# - Return the metrics collector as JSON
@app.get("/metrics")
async def get_metrics():
    pass
