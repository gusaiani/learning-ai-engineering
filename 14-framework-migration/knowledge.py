"""
Knowledge base: ingest documents, embed chunks, search with RAG.

Usage:
    python knowledge.py ingest <path>      # Ingest a file or directory
    python knowledge.py search <query>     # Semantic search
    python knowledge.py list               # Show ingested documents
"""

import argparse
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import chroma_client, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, observe

# ---------------------------------------------------------------------------
# ChromaDB collection
# ---------------------------------------------------------------------------

COLLECTION_NAME = "knowledge_base"

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSIONS,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    client=chroma_client, # reuse the chromabd client from config — no double-lock
)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by character count.

    Returns a list of strings, each at most chunk_size characters,
    with `overlap` characters shared between consecutive chunks.
    Empty input returns an empty list.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
@observe(name="embed_texts")
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Return list of vectors."""
    return embeddings.embed_documents(texts)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_file(file_path: Path) -> dict:
    """Read a file, chunk it, embed chunks, store in ChromaDB.

    Returns: {"file": str, "chunks": int, "status": "ok"|"error", "error": str|None}
    """
    try:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            return {"file": file_path.name, "chunks": 0, "status": "ok", "error": None}

        ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]

        vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        return {"file": file_path.name, "chunks": len(chunks), "status": "ok", "error": None}
    except Exception as e:
        return {"file": file_path.name, "chunks": 0, "status": "error", "error": str(e)}


def ingest_directory(dir_path: Path) -> list[dict]:
    """Ingest all .md and .txt files in a directory."""
    results = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix in (".md", ".txt"):
            results.append(ingest_file(f))
    return results


# ---------------------------------------------------------------------------
# Search
# -----------------
@observe(name="search")
def search(query: str, top_k: int = 5) -> list[dict]:
    """Embed query, search ChromaDB, return top-k results.

    Each result: {"text": str, "source": str, "score": float, "chunk_index": int}
    """
    matches = vectorstore.similarity_search_with_score(query, k=top_k)

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "score": score,
            "chunk_index": doc.metadata["chunk_index"],
        }
        for doc, score in matches
    ]


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_documents() -> dict:
    """Return a summary of what's in the knowledge base.

    Returns: {"total_chunks": int, "sources": list[str]}
    """
    all_data = vectorstore.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    if not metadatas:
        return {"total_chunks": 0, "sources": []}

    sources = sorted({m["source"] for m in metadatas})
    return {"total_chunks": len(metadatas), "sources": sources}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Knowledge base manager")
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest", help="Ingest a file or directory")
    ingest_p.add_argument("path", type=Path)

    search_p = sub.add_parser("search", help="Semantic search")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=5)

    sub.add_parser("list", help="List knowledge base contents")

    args = parser.parse_args()

    if args.command == "ingest":
        path = args.path
        if path.is_dir():
            results = ingest_directory(path)
        elif path.is_file():
            results = [ingest_file(path)]
        else:
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        for r in results:
            status = "ok" if r["status"] == "ok" else "FAIL"
            print(f"  [{status}] {r['file']} ({r['chunks']} chunks)")
            if r.get("error"):
                print(f"        {r['error']}")

    elif args.command == "search":
        results = search(args.query, top_k=args.top_k)
        if not results:
            print("No results. Have you ingested documents?")
            return
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['score']:.3f}, source: {r['source']}) ---")
            print(r["text"][:300])

    elif args.command == "list":
        info = list_documents()
        print(f"Total chunks: {info['total_chunks']}")
        if info["sources"]:
            print(f"Sources: {', '.join(info['sources'])}")
        else:
            print("Knowledge base is empty.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
