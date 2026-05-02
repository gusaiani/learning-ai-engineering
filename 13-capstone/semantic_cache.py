"""
Semantic response cache.

On cache hit, the full agent loop is skipped and the cached response is replayed.
Only used for stateless requests (no session_id / customer_id / image_path) — contextual requests can have different answers for the same query.

Match: cosine similarity above SIMILARITY_THRESHOLD against any stored embedding.
TTL: entries older than TTL_SECONDS are ignored (and pruned on next access).

In-memory only — restarts wipe the cache. Swap for Redis/pgvector in prod.
"""

import time

from knowledge import embed_texts

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

def lookup(query: str) -> str | None:
    """Return a cached response for a semantically-similar query, or None on miss."""
    query_embedding = embed_texts([query])[0]
    now = time.time()

    best_response = None
    best_similarity = SIMILARITY_THRESHOLD

    for embedding, _stored_query, response, stored_at in _ENTRIES:
        if now - stored_at > TTL_SECONDS:
            continue # expired
        similarity = _cosine_similarity(query_embedding, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = response

    return best_response

def store(query: str, response: str) -> None:
    """Embed the query and append (embedding, query, response, ts) to the cache.
    Also prunes any expired entries while we're here."""
    now = time.time()

    # Prune expired entries before adding the new one — keeps the list bounded.
    _ENTRIES[:] = [
        entry for entry in _ENTRIES
        if now - entry[3] <= TTL_SECONDS
    ]

    embedding = embed_texts([query])[0]
    _ENTRIES.append((embedding, query, response, now))
