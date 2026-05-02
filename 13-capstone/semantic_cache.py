"""
Semantic response cache.

On cache hit, the full agent loop is skipped and the cached response is replayed.
Only used for stateless requests (no session_id / customer_id / image_path) — contextual requests can have different answers for the same query.

Match: cosine similarity above SIMILARITY_THRESHOLD against any stored embedding.
TTL: entries older than TTL_SECONDS are ignored (and pruned on next access).

In-memory only — restarts wipe the cache. Swap for Redis/pgvector in prod.
"""

import time

# Tunables
SIMILARITY_THRESHOLD = 0.95  # cosine sim above this counts as a hit
TTL_SECONDS = 3600           # 1 hour

# (embedding, query_text, response_text, stored_at_ts)
_ENTRIES: list[tuple[list[float], str, str, float]] = []

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns a value in [-1, 1]."""
    dot = sum (x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)
