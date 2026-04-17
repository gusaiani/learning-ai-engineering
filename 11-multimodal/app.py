"""
Document Intelligence CLI

Usage:
    python app.py extract <image>
    python app.py transcribe <audio>
    python app.py analyze <image> [--notes <audio>]
    python app.py list
    python app.py ask <question>
"""

import argparse
import base64
import json
import mimetypes
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Clients & constants
# ---------------------------------------------------------------------------

client = OpenAI()

VISION_MODEL = "gpt-4o"
CHAT_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"

DOCS_PATH = Path(__file__).parent / "docs.json"

# Prices as of early 2025 — update if models change
MODEL_PRICES = {
    "gpt-4o":      {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}
WHISPER_PRICE_PER_MIN = 0.006


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    return input_tokens * prices["input"] + output_tokens * prices["output"]


# ---------------------------------------------------------------------------
# Extraction schema
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    description: str
    amount: float | None = None
    quantity: float | None = None


class KeyValue(BaseModel):
    name: str
    value: str


class ExtractedDocument(BaseModel):
    document_type: Literal["receipt", "invoice", "id", "business_card", "form", "letter", "other"]
    vendor: str | None = Field(None, description="Name of the issuing business or person, if visible")
    date: str | None = Field(None, description="Document date in ISO format if visible")
    total: float | None = Field(None, description="Grand total amount, if applicable")
    currency: str | None = Field(None, description="ISO currency code if visible (USD, EUR, BRL...)")
    line_items: list[LineItem] = Field(default_factory=list)
    key_fields: list[KeyValue] = Field(
        default_factory=list,
        description="Other relevant fields extracted as name/value pairs",
    )
    full_text: str = Field(description="Complete OCR-style transcription of all visible text")
    notes_summary: str | None = Field(
        None,
        description="If the user provided a spoken note, a one-sentence summary of its relevance",
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load_docs() -> list[dict]:
    if not DOCS_PATH.exists():
        return []
    return json.loads(DOCS_PATH.read_text())


def save_docs(docs: list[dict]) -> None:
    DOCS_PATH.write_text(json.dumps(docs, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

# - Read the file as bytes
# - Detect the MIME type with mimetypes.guess_type (fall back to "image/jpeg")
# - Base64-encode the bytes and decode to a str
# - Return a data URL: f"data:{mime};base64,{b64}"
def encode_image(path: Path) -> str:
    image_bytes = path.read_bytes()
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------------------
# Vision extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = (
    "You are a document intelligence assistant. You extract structured data from "
    "images of real-world documents (receipts, invoices, IDs, forms, letters). "
    "Read every visible field carefully. If a field isn't visible, leave it null "
    "rather than guessing. Always include the complete visible text in `full_text`."
)


def extract_document(
    image_path: Path,
    note_transcript: str | None = None,
) -> tuple[ExtractedDocument, dict]:
    data_url = encode_image(image_path)

    USER_INSTRUCTION = (
        "Extract all fields you can read from this document. "
        "Leave fields null if they are not visible. "
        "Transcribe the complete visible text into `full_text`."
    )

    user_content = [
        {"type": "text", "text": USER_INSTRUCTION},
        {
            "type": "image_url", 
            "image_url": {"url": data_url, "detail": "high"},
        }
    ]

    if note_transcript:
        user_content.append(
            {"type": "text", "text": f"Spoken note from the user: {note_transcript}"}
        )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = client.beta.chat.completions.parse(
        model=VISION_MODEL,
        messages=messages,
        response_format=ExtractedDocument,
    )

    extracted = response.choices[0].message.parsed
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    cost = calculate_cost(VISION_MODEL, input_tokens, output_tokens)

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }

    return extracted, usage


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: Path) -> tuple[str, dict]:
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
        )

    cost = (result.duration / 60) * WHISPER_PRICE_PER_MIN
    info = {"duration": result.duration, "cost": cost}
    return result.text, info

# ---------------------------------------------------------------------------
# Pipeline: analyze
# ---------------------------------------------------------------------------


def analyze(image_path: Path, notes_path: Path | None = None) -> dict:
    transcript = None
    audio_usage = None
    
    if notes_path:
        transcript, audio_usage = transcribe_audio(notes_path)
    
    extracted, vision_usage = extract_document(image_path, transcript)

    audio_cost = audio_usage["cost"] if audio_usage else 0
    total_cost = round(vision_usage["cost"] + audio_cost, 6)

    record = {
        "id": uuid.uuid4().hex[:8],
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "source_image": str(image_path),
        "source_audio": str(notes_path) if notes_path else None,
        "note_transcript": transcript,
        "extracted": extracted.model_dump(),
        "cost": {
            "vision": vision_usage,
            "whisper": audio_usage,
            "total_usd": total_cost,
        },
    }

    docs = load_docs()
    docs.append(record)
    save_docs(docs)

    return record

# ---------------------------------------------------------------------------
# Q&A over stored documents
# ---------------------------------------------------------------------------

ASK_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a user's personal document "
    "library. You are given a list of previously extracted documents. Answer "
    "based only on what's in the library. If the library doesn't contain the "
    "answer, say so clearly."
)


def format_doc_for_context(record: dict) -> str:
    """Compact one-block summary used when building the Q&A context."""
    ex = record["extracted"]
    header = (
        f"[id={record['id']} | {record['created_at']} | {ex['document_type']}"
        + (f" | vendor={ex['vendor']}" if ex.get("vendor") else "")
        + (f" | total={ex['total']} {ex.get('currency') or ''}".rstrip() if ex.get("total") is not None else "")
        + "]"
    )
    parts = [header]
    if record.get("note_transcript"):
        parts.append(f"Notes: {record['note_transcript']}")
    if ex.get("key_fields"):
        parts.append("Fields: " + ", ".join(f"{kv['name']}={kv['value']}" for kv in ex["key_fields"]))
    if ex.get("full_text"):
        parts.append(f"Text: {ex['full_text']}")
    return "\n".join(parts)


def ask(question: str) -> None:
    docs = load_docs()
    if not docs:
        print("No documents in library yet. Run `analyze` first.")
        return

    context = "\n\n".join(format_doc_for_context(r) for r in docs)

    system_content = ASK_SYSTEM_PROMPT + "\n\nLibrary:\n" + context

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    answer = response.choices[0].message.content
    print(answer)

# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def list_docs() -> None:
    docs = load_docs()
    if not docs:
        print("No documents yet. Run `analyze` first.")
        return
    for r in docs:
        ex = r["extracted"]
        total_str = ""
        if ex.get("total") is not None:
            total_str = f"{ex['total']:.2f} {ex.get('currency') or ''}".strip()
        print(
            f"{r['id']}  {r['created_at']}  {ex['document_type']:<13}  "
            f"{(ex.get('vendor') or '—'):<25}  {total_str}"
        )


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Document intelligence CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract structured data from an image")
    p_extract.add_argument("image", type=Path)

    p_trans = sub.add_parser("transcribe", help="Transcribe an audio file")
    p_trans.add_argument("audio", type=Path)

    p_analyze = sub.add_parser("analyze", help="Extract + optionally transcribe, save to docs.json")
    p_analyze.add_argument("image", type=Path)
    p_analyze.add_argument("--notes", type=Path, default=None, help="Optional audio note")

    sub.add_parser("list", help="List saved documents")

    p_ask = sub.add_parser("ask", help="Ask a question across saved documents")
    p_ask.add_argument("question", nargs="+")

    return p


def main() -> None:
    args = build_parser().parse_args()

    try:
        if args.cmd == "extract":
            extracted, usage = extract_document(args.image)
            print(json.dumps(extracted.model_dump(), indent=2, ensure_ascii=False))
            print(f"\ntokens: {usage['input_tokens']} in / {usage['output_tokens']} out  cost: ${usage['cost']:.4f}")
        elif args.cmd == "transcribe":
            text, info = transcribe_audio(args.audio)
            print(text)
            print(f"\nduration: {info['duration']:.1f}s  cost: ${info['cost']:.4f}")
        elif args.cmd == "analyze":
            record = analyze(args.image, args.notes)
            print(json.dumps(record, indent=2, ensure_ascii=False))
        elif args.cmd == "list":
            list_docs()
        elif args.cmd == "ask":
            ask(" ".join(args.question))
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Not implemented yet: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
