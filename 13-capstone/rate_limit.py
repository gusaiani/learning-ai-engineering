"""
Token-bucket rate limiting per customer (or anonymous global bucket).

Each key gets a bucket with CAPACITY tokens that refills at REFILL_RATE tokens per second. Each request consumes one token. When the bucket is empty, requests are rejected with HTTP 429.

In-memory only — restarts reset all buckets. Swap for Redis in prod.
"""

import time

from fastapi import HTTPException

# Tunables — generous enough for testing, tight enough to actually trip
CAPACITY = 5      # max burst size (tokens)
REFILL_RATE = 0.5 # tokens added per second (=30/min sustained)

# key -> (tokens_remaining, last_refill_timestamp)
_BUCKETS: dict[str, tuple[float, float]] = {}

def _take_token(key: str) -> bool:
    """Try to consume one token from the bucket. Return True if allowed, False if rate-limited."""
    now = time.time()
    tokens, last_refill = _BUCKETS.get(key, (CAPACITY, now))

    # Refill: add tokens proportional to elapsed time, capped at CAPACITY
    elapsed = now - last_refill
    tokens = min(CAPACITY, tokens + elapsed * REFILL_RATE)

    if tokens < 1:
        _BUCKETS[key] = (tokens, now)
        return False

    _BUCKETS[key] = (tokens - 1, now)
    return True

def check_rate_limit(customer_id: str | None) -> None:
    """Raise HTTP 429 if the caller has exhausted their budget."""
    key = customer_id or "anonymous"
    if not _take_token(key):
        retry_after = int(1 / REFILL_RATE)  # seconds until the next token
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {key}. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )