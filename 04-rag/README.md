# Module 04 — RAG (Retrieval-Augmented Generation)

**Goal:** Build a Q&A system that answers questions grounded in a PDF corpus — not from the model's training data.

**Time:** ~2 days

---

## Setup & running

```bash
pip install anthropic openai python-dotenv pymupdf numpy rich

# Index your PDFs (first run builds the vector index)
python rag.py "What is the main argument of the paper?"

# Force re-index after adding new PDFs
python rag.py --reindex "What is the main argument?"
```

Place PDF files in the `pdfs/` folder before running.

---

## What you'll learn

- How RAG works end-to-end: ingest → chunk → embed → retrieve → generate
- Why chunking strategy matters more than model choice
- How to extract text from PDFs programmatically
- How to build a simple but effective retrieval pipeline
- How to prompt an LLM to answer only from retrieved context
- The difference between naive RAG and production RAG

---

## Concepts

### What is RAG?

RAG = Retrieval-Augmented Generation. Instead of asking the LLM to answer from memory, you:

1. **Retrieve** relevant chunks of text from your documents
2. **Augment** the prompt with those chunks as context
3. **Generate** an answer grounded in the retrieved text

```
User question
    ↓
Embed question → find similar chunks → stuff into prompt → LLM answers
```

This solves three big problems:
- **Hallucination:** The model answers from your data, not its imagination
- **Freshness:** Your docs can be updated without retraining
- **Attribution:** You can show which chunks the answer came from

### Chunking

You can't embed an entire PDF as one vector — it's too long and the embedding loses specificity. Instead, you split documents into **chunks**.

```
Document → [chunk_1, chunk_2, chunk_3, ...]
```

Key decisions:
- **Chunk size:** 200–500 tokens is a sweet spot. Too small = no context. Too large = diluted meaning.
- **Overlap:** Overlapping chunks (e.g., 50-token overlap) prevent splitting important info across chunk boundaries.
- **Boundaries:** Splitting on paragraphs or sentences is better than splitting mid-sentence.

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
```

### The retrieval step

Same as Module 03's semantic search, but now over chunks instead of whole documents:

1. Embed all chunks at index time
2. Embed the user's query at search time
3. Compute cosine similarity
4. Return the top-k most relevant chunks

### The generation step

Take the retrieved chunks and stuff them into a prompt:

```
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{chunk_1}
{chunk_2}
{chunk_3}

Question: {user_question}
```

This is called **context stuffing** — the simplest RAG generation strategy.

### Why RAG beats long-context alone

Even models with 100k+ token windows benefit from RAG:
- **Cost:** Sending 100 pages every query is expensive. RAG sends only the relevant chunks.
- **Accuracy:** Models attend better to shorter, relevant context than to a huge document.
- **Scale:** You can index thousands of documents. No context window is big enough for that.

---

## Project: Q&A over a PDF Corpus

Build a CLI tool that:

1. Loads all PDFs from a `pdfs/` folder
2. Extracts text and chunks each document
3. Embeds all chunks (with caching)
4. Takes a question from the command line
5. Retrieves the most relevant chunks
6. Sends them to Claude with the question
7. Prints the answer with source citations

### Requirements

```
- Extract text from PDFs using PyMuPDF (fitz)
- Chunk text into ~400-word segments with 50-word overlap
- Embed chunks using text-embedding-3-small (reuse from Module 03)
- Cache the chunk index to chunks_index.json
- Retrieve top-5 chunks by cosine similarity
- Send retrieved chunks + question to Claude (claude-sonnet-4-20250514)
- Print the answer and list which PDF(s) / chunk(s) it came from
- Handle the case where no relevant chunks are found
```

### Starter code

```python
# rag.py
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
    # TODO: open with fitz.open(), iterate pages, extract text
    pass


# ── Chunking ────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    Returns a list of chunk strings.
    """
    # TODO: split on whitespace, slide a window with overlap
    pass


# ── Embedding ───────────────────────────────────────────────────


def embed(text: str) -> list[float]:
    """Return the embedding vector for a string using OpenAI."""
    # TODO: reuse your Module 03 approach
    pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    # TODO: reuse from Module 03
    pass


# ── Indexing ────────────────────────────────────────────────────


def build_index(force: bool = False) -> list[dict]:
    """
    Build or load the chunk index.

    Each entry in the index is a dict:
        {"source": "filename.pdf", "chunk_id": 0, "text": "...", "embedding": [...]}

    If CACHE_FILE exists and force=False, load from cache.
    Otherwise, extract text from all PDFs, chunk, embed, and save.
    """
    # TODO: implement
    pass


# ── Retrieval ───────────────────────────────────────────────────


def retrieve(query: str, index: list[dict], top_k: int = 5) -> list[dict]:
    """
    Embed the query, compute similarity against all chunks,
    return the top_k most relevant chunks (as dicts with score added).
    """
    # TODO: implement
    pass


# ── Generation ──────────────────────────────────────────────────


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Send the query + retrieved chunks to Claude.
    The prompt should instruct the model to answer ONLY from the provided context.
    Returns the model's answer as a string.
    """
    # TODO: build a prompt with context, call anthropic.messages.create()
    pass


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
```

### Your task

1. **`extract_text_from_pdf()`** — Use `fitz.open()` to iterate pages and extract text
2. **`chunk_text()`** — Split text into overlapping word-based chunks
3. **`embed()`** and **`cosine_similarity()`** — Reuse from Module 03
4. **`build_index()`** — Extract → chunk → embed all PDFs, save to JSON cache
5. **`retrieve()`** — Embed query, score chunks, return top-k with scores
6. **`generate_answer()`** — Build a grounded prompt and call Claude

### Sample PDFs

Drop 2–3 PDFs into `pdfs/`. Good options:
- Academic papers from arXiv (e.g., "Attention Is All You Need")
- Any public-domain report or whitepaper
- Your own notes or documents

---

## Stretch goals

- Add **reranking**: after retrieval, use a second LLM call to rerank chunks by relevance before generation
- Show a **confidence score** based on the highest chunk similarity
- Support **multi-turn conversation** — let the user ask follow-up questions with context
- Add a `--verbose` flag to show the full prompt sent to Claude
- Chunk by **paragraphs or sentences** instead of word count, and compare quality
- Add **metadata filtering** (e.g., only search chunks from a specific PDF)

---

## Key questions to answer before moving on

1. What happens if your chunks are too small (e.g., 50 words)? Too large (e.g., 2000 words)?
2. Why do we embed with OpenAI but generate with Claude? Could you use the same provider for both?
3. How would you handle a question that spans information across multiple documents?
4. What are the failure modes of naive RAG? (Hint: think about what the retrieval step can get wrong)
5. How would you evaluate whether your RAG system is giving good answers?

---

## Resources

- [Anthropic RAG guide](https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation)
- [PyMuPDF docs](https://pymupdf.readthedocs.io/)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [Chunking strategies — Pinecone](https://www.pinecone.io/learn/chunking-strategies/)

---

**When done:** Mark Module 04 as shipped in the root README, commit, and move to [Module 05](../05-agents-tool-use/).
