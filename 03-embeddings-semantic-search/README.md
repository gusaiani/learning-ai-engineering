# Module 03 — Embeddings & Semantic Search

**Goal:** Understand what embeddings are, how similarity search works, and build a semantic search engine over a document collection.

**Time:** ~1 day

---

## Setup & running

```bash
pip install openai python-dotenv rich numpy

python search.py "your query here"
```

---

## What you'll learn

- What an embedding is and why it's useful
- How cosine similarity measures semantic closeness
- How to embed documents and queries with the OpenAI API
- How to build an in-memory semantic search without a vector DB
- The trade-offs between keyword search and semantic search

---

## Concepts

### What is an embedding?

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meaning end up close together in vector space — even if they use different words.

```python
"dog"  → [0.12, -0.83, 0.41, ...]  # 1536 numbers
"puppy" → [0.14, -0.79, 0.38, ...]  # very close!
"car"   → [-0.62, 0.31, -0.55, ...] # far away
```

### Cosine similarity

The standard way to measure closeness between two vectors. Returns a value from -1 (opposite) to 1 (identical).

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

You don't need to understand the math deeply — just know: **higher = more similar**.

### The embedding pipeline

```
Documents → embed each → store vectors
Query     → embed      → compare to all stored vectors → return top-k
```

### OpenAI embedding model

`text-embedding-3-small` is fast and cheap. It outputs 1536-dimensional vectors.

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="some text here",
)
vector = response.data[0].embedding  # list of 1536 floats
```

---

## Project: Semantic Search over Documents

Build a CLI tool that:

1. Loads a collection of text documents from a folder
2. Embeds all documents (cache embeddings to avoid re-computing)
3. Takes a query from the command line
4. Returns the top-k most semantically similar documents

### Requirements

```
- Load .txt files from a docs/ folder
- Embed each document using text-embedding-3-small
- Cache embeddings to embeddings.json (skip re-embedding on next run)
- Accept a query as a CLI argument
- Compute cosine similarity between query and all docs
- Print top-3 results with their similarity scores
```

### Starter code

```python
# search.py
import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv(Path(__file__).parent.parent / ".env")
client = OpenAI()
DOCS_DIR = Path(__file__).parent / "docs"
CACHE_FILE = Path(__file__).parent / "embeddings.json"
EMBED_MODEL = "text-embedding-3-small"


def embed(text: str) -> list[float]:
    """Return the embedding vector for a string."""
    # TODO: call client.embeddings.create() and return the vector
    pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    # TODO: implement using numpy
    pass


def load_docs() -> dict[str, str]:
    """Load all .txt files from DOCS_DIR. Returns {filename: content}."""
    # TODO: implement
    pass


def load_or_build_index() -> dict[str, list[float]]:
    """
    Load cached embeddings from CACHE_FILE if it exists,
    otherwise embed all docs and save to cache.
    Returns {filename: embedding_vector}.
    """
    # TODO: implement
    pass


def search(query: str, index: dict[str, list[float]], top_k: int = 3) -> list[tuple[str, float]]:
    """
    Embed the query, compute cosine similarity against all docs,
    return top_k results as [(filename, score)].
    """
    # TODO: implement
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    index = load_or_build_index()
    results = search(query, index)

    for filename, score in results:
        print(f"{score:.3f}  {filename}")
```

### Your task

1. Implement `embed()` — one API call, return `response.data[0].embedding`
2. Implement `cosine_similarity()` — use numpy dot product and norms
3. Implement `load_docs()` — glob for `*.txt` files, read each
4. Implement `load_or_build_index()` — check if cache exists, load it; otherwise embed all docs and save
5. Implement `search()` — embed query, score all docs, sort descending, return top-k

### Sample docs to test with

Create a `docs/` folder and add a few `.txt` files on different topics. A mix of tech, science, and random topics works well to test semantic search.

---

## Stretch goals

- Use `rich` to print results as a table with highlighted scores
- Add a `--top` CLI flag to control how many results to show
- Add a `--reindex` flag to force re-embedding even if cache exists
- Try `text-embedding-3-large` and compare result quality

---

## Key questions to answer before moving on

1. Why is caching embeddings important in a real application?
2. What happens to search quality if you embed very long documents as a single chunk?
3. Why does semantic search sometimes return worse results than keyword search?
4. What's the difference between `text-embedding-3-small` and `text-embedding-3-large` beyond size?

---

## Resources

- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [text-embedding-3-small model card](https://platform.openai.com/docs/models)
- [Cosine similarity — Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)

---

**When done:** Mark Module 03 as shipped in the root README, commit, and move to [Module 04](../04-rag/).
