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
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_docs() -> dict[str, str]:
    """Load all .txt files from DOCS_DIR. Returns {filename: content}."""
    return {f.name: f.read_text() for f in DOCS_DIR.glob("*.txt")}


def load_or_build_index() -> dict[str, list[float]]:
    """
    Load cached embeddings from CACHE_FILE if it exists,
    otherwise embed all docs and save to cache.
    Returns {filename: embedding_vector}.
    """
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())

    docs = load_docs()
    index = {name: embed(content) for name, content in docs.items()}
    CACHE_FILE.write_text(json.dumps(index))
    return index


def search(query: str, index: dict[str, list[float]], top_k: int = 3) -> list[tuple[str, float]]:
    """
    Embed the query, compute cosine similarity against all docs,
    return top_k results as [(filename, score)].
    """
    query_vec = embed(query)
    scores = [(name, cosine_similarity(query_vec, vec)) for name, vec in index.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    index = load_or_build_index()
    results = search(query, index)

    for filename, score in results:
        print(f"{score:.3f}  {filename}")
