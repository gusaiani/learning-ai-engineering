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
