"""
Shared configuration: clients, models, pricing, and observability setup.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

openai_client = OpenAI()

CHROMA_PATH = Path(__file__).parent / ".chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

try:
    from langfuse import Langfuse, observe

    langfuse_client = Langfuse()
except Exception:
    langfuse_client = None

    def observe(*args, **kwargs):
        """No-op decorator when Langfuse is not configured"""
        def decorator(fn):
            return fn
        return decorator

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512

# ---------------------------------------------------------------------------
# Pricing (per token, as of early 2025)
# ---------------------------------------------------------------------------

MODEL_PRICES = {
    "gpt-4o":                 {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":            {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a given model and token counts."""
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    return input_tokens * prices["input"] + output_tokens * prices["output"]
