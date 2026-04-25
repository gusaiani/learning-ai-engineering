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
    start = time.time()

    async def event_generator():
        for event in run_support_agent(req.message, req.customer_id, req.image_path):
            if event.type == "done":
                latency_ms = (time.time() - start) * 1000
                _metrics["total_requests"] += 1
                _metrics["total_latency_ms"] += latency_ms
            elif event.type == "error":
                _metrics["total_errors"] += 1

            yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest):
    """Non-streaming chat. Collects all events and returns a single response."""
    start = time.time()

    response_parts = []
    tools_called = []
    routing = {}
    cost = 0.0

    for event in run_support_agent(req.message, req.customer_id, req.image_path):
        if event.type == "token":
            response_parts.append(event.data["content"])
        elif event.type == "status":
            if "tool" in event.data and event.data.get("status") == "calling":
                tools_called.append(event.data["tool"])
            if "route" in event.data:
                routing = event.data
        elif event.type == "error":
            _metrics["total_errors"] += 1
            raise HTTPException(status_code=500, detail=event.data["message"])

    latency_ms = (time.time() - start) * 1000
    _metrics["total_requests"] += 1
    _metrics["total_latency_ms"] += latency_ms
    _metrics["total_cost"] += cost

    return ChatResponse(
        response="".join(response_parts),
        routing=routing,
        tools_called=tools_called,
        cost=cost,
        latency_ms=round(latency_ms, 1),
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest a document or directory into the knowledge base."""
    path = Path(req.path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    if path.is_dir():
        results = ingest_directory(path)
    else:
        results = [ingest_file(path)]

    return IngestResponse(results=results)


@app.get("/knowledge")
async def knowledge():
    """List knowledge base contents."""
    return list_documents()


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Operational metrics."""
    total = _metrics["total_requests"]
    avg_latency = _metrics["total_latency_ms"] / total if total else 0.0

    return MetricsResponse(
        total_requests=total,
        avg_latency_ms=round(avg_latency, 1),
        total_cost=_metrics["total_cost"],
        total_errors=_metrics["total_errors"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _start_time, 1)}
