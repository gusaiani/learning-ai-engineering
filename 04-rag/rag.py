# rag.py — RAG Q&A over a PDF corpus
import os
import sys
import json
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

anthropic = Anthropic()
openai_client = OpenAI()

PDFS_DIR = Path(__file__).parent / "pdfs"
CACHE_FILE = Path(__file__).parent / "chunks_index.json"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-20250514"
CHUNK_SIZE = 400  # words
CHUNK_OVERLAP = 50  # words


# ── PDF extraction ──────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# ── Chunking ────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── Embedding ───────────────────────────────────────────────────


def embed(text: str) -> list[float]:
    """Return the embedding vector for a string using OpenAI."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input = text)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Indexing ────────────────────────────────────────────────────


def build_index(force: bool = False) -> list[dict]:
    """
    Build or load the chunk index.

    Each entry in the index is a dict:
        {"source": "filename.pdf", "chunk_id": 0, "text": "...", "embedding": [...]}

    If CACHE_FILE exists and force=False, load from cache.
    Otherwise, extract text from all PDFs, chunk, embed, and save.
    """
    if CACHE_FILE.exists() and not force:
        return json.loads(CACHE_FILE.read_text())

    chunks = []
    for pdf_path in PDFS_DIR.glob("*.pdf"):
        text = extract_text_from_pdf(pdf_path)
        for i, text_chunk in enumerate(chunk_text(text)):
            chunks.append({"source": pdf_path.name, "chunk_id": i, "text": text_chunk})

    for chunk in chunks:
        chunk["embedding"] = embed(chunk["text"])

    CACHE_FILE.write_text(json.dumps(chunks))
    return chunks


# ── Retrieval ───────────────────────────────────────────────────


def retrieve(query: str, index: list[dict], top_k: int = 5) -> list[dict]:
    """
    Embed the query, compute similarity against all chunks,
    return the top_k most relevant chunks (as dicts with score added).
    """
    query_vec = embed(query)
    for chunk in index:
        chunk["score"] = cosine_similarity(query_vec, chunk["embedding"])
    ranked = sorted(index, key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]


# ── Generation ──────────────────────────────────────────────────


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Send the query + retrieved chunks to Claude.
    The prompt should instruct the model to answer ONLY from the provided context.
    Returns the model's answer as a string.
    """
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    response = anthropic.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system="Answer using ONLY the provided context. If the context doesn't contain enough information, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return response.content[0].text


# ── CLI ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Q&A over PDFs")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    index = build_index(force=args.reindex)
    chunks = retrieve(args.query, index, top_k=args.top_k)

    if not chunks:
        print("No relevant chunks found.")
        sys.exit(1)

    print(f"\n📚 Retrieved {len(chunks)} chunks from: {', '.join(set(c['source'] for c in chunks))}\n")

    answer = generate_answer(args.query, chunks)
    print(answer)

    print("\n--- Sources ---")
    for c in chunks:
        print(f"  [{c['source']} chunk {c['chunk_id']}] (score: {c['score']:.3f})")
