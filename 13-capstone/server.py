"""
FastAPI server: streaming chat, document ingestion, metrics.

Usage:
    uvicorn server:app --reload --port 8000
"""

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import run_support_agent, AgentEvent
from knowledge import ingest_file, ingest_directory, list_documents, search

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="NovaCRM Support Agent", version="1.0.0")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    routing: dict
    tools_called: list[str]
    cost: float
    latency_ms: float


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    results: list[dict]


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    total_cost: float
    total_errors: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# In-memory metrics
# ---------------------------------------------------------------------------

_start_time = time.time()
_metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "total_cost": 0.0,
    "total_errors": 0,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint. Returns Server-Sent Events.

    Event types:
      event: status   — tool call or routing info
      event: token    — a piece of the response text
      event: done     — request complete, includes cost/usage
      event: error    — something went wrong
    """
    # TODO 1: Implement streaming SSE endpoint.
    #
    # 1. Record start time
    # 2. Create a sync generator from run_support_agent(req.message, req.customer_id, req.image_path)
    # 3. Write an async generator that:
    #    a. Iterates over AgentEvent objects
    #    b. Formats each as SSE: f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"
    #    c. On "done" event, update _metrics (total_requests, total_latency_ms, total_cost)
    #    d. On "error" event, increment _metrics["total_errors"]
    # 4. Return StreamingResponse(generator, media_type="text/event-stream")
    #
    # Note: run_support_agent returns a sync generator. You can iterate it
    # in an async generator with a regular for loop (FastAPI handles threading).
    raise NotImplementedError


@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest):
    """Non-streaming chat. Collects all events and returns a single response."""
    # TODO 2: Implement non-streaming endpoint.
    #
    # 1. Record start time
    # 2. Iterate over run_support_agent() events, collecting:
    #    - All "token" events → concatenate into response string
    #    - All "status" events with tool info → collect tool names
    #    - "status" event with "route" → save as routing info
    #    - "done" event → extract cost
    # 3. Calculate latency
    # 4. Update _metrics
    # 5. Return ChatResponse(...)
    raise NotImplementedError


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest a document or directory into the knowledge base."""
    # TODO 3: Detect if req.path is a file or directory.
    # Call ingest_file() or ingest_directory() accordingly.
    # Return IngestResponse with the results.
    # Raise HTTPException(404) if the path doesn't exist.
    raise NotImplementedError


@app.get("/knowledge")
async def knowledge():
    """List knowledge base contents."""
    # TODO 4: Call list_documents() and return the result.
    raise NotImplementedError


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Operational metrics."""
    # TODO 5: Calculate avg_latency and uptime, return MetricsResponse.
    #
    # avg_latency_ms = total_latency / total_requests (or 0 if no requests)
    # uptime_seconds = time.time() - _start_time
    raise NotImplementedError


@app.get("/health")
async def health():
    """Health check."""
    # TODO 6: Return {"status": "ok", "uptime_seconds": time.time() - _start_time}
    raise NotImplementedError
