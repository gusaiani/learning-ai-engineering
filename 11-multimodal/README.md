# Module 11 — Multimodal: Vision & Audio

**Goal:** Move past plain text. Build a tool that ingests images of real documents, transcribes spoken notes, and extracts structured data you can query.

**Time:** ~2 days

---

## Setup & running

```bash
pip install openai pydantic python-dotenv pillow

# Extract structured data from a single image
python app.py extract samples/receipt.jpg

# Transcribe an audio note
python app.py transcribe samples/note.m4a

# Full pipeline: image + spoken notes → one record saved to docs.json
python app.py analyze samples/receipt.jpg --notes samples/note.m4a

# List all records you've ingested
python app.py list

# Ask a question across all ingested documents
python app.py ask "What did I spend on coffee last month?"
```

You'll need an `OPENAI_API_KEY` in `.env`.

---

## What you'll learn

- How vision-capable LLMs "see" images (they don't — they tokenize them)
- The real cost of sending an image: how image tokens are counted
- When to use a general vision model (GPT-4o) vs a dedicated OCR tool (Tesseract, Textract)
- Structured data extraction from images using Pydantic + JSON schema
- Whisper transcription: formats, language hints, word-level timestamps
- Combining modalities: image + audio → unified record
- Simple file-based storage for a personal document library

---

## Concepts

### What "multimodal" actually means

Until this module every input you've sent to an LLM has been text. Multimodal models accept other input types — images, audio, video — and tokenize them into the same latent space as text. From the model's perspective, an image is just another prefix of tokens before your question.

Two things are worth internalizing:

1. **Images cost tokens.** A high-detail 1024×1024 image on GPT-4o costs around 765 tokens. A whole receipt photo can be 1,000+ tokens before you've typed a single character. Budget accordingly.
2. **The model doesn't "look twice".** It sees the image once, during its forward pass. If you ask follow-up questions about fine details, it's working from whatever representation it built the first time — not re-scanning pixels. So if you need precision, extract everything you'll ever want in one call.

### How to send an image to GPT-4o

Two forms, both via the `messages` array:

```python
# Base64 (private files on your machine)
{
  "type": "image_url",
  "image_url": {
    "url": f"data:image/jpeg;base64,{b64_encoded_bytes}",
    "detail": "high"  # "low" is cheaper but lossy; "auto" lets OpenAI decide
  }
}

# Public URL
{
  "type": "image_url",
  "image_url": {"url": "https://example.com/receipt.jpg", "detail": "high"}
}
```

`detail` is worth knowing:

| Level | Tokens (approx.) | When to use |
|-------|------------------|-------------|
| `low` | 85 | Fast scan, rough classification, thumbnails |
| `high` | 765 + 170 per 512-px tile | Reading text, extracting data, anything precise |
| `auto` | Either | When you don't know the image size in advance |

For a document-intelligence tool, you almost always want `high` — the whole point is reading small text.

### Structured extraction: don't parse free-form text

A beginner version of this project would prompt the model to "describe the receipt" and then regex the result. Don't. Use **structured outputs** — OpenAI's JSON-schema-enforced response format that guarantees the model returns data matching your Pydantic model:

```python
from pydantic import BaseModel

class ExtractedDocument(BaseModel):
    document_type: Literal["receipt", "invoice", "id", "form", "other"]
    vendor: str | None
    total: float | None
    line_items: list[LineItem]
    full_text: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=ExtractedDocument,
)
extracted: ExtractedDocument = response.choices[0].message.parsed
```

No parsing, no retries, no `json.loads` try/except. The model either returns valid `ExtractedDocument` or the SDK raises.

### Vision model vs dedicated OCR

Dedicated OCR (Tesseract, AWS Textract, Google Document AI) has been around for decades. Why reach for a general LLM?

| | Tesseract / Textract | GPT-4o Vision |
|---|---|---|
| Raw text accuracy on clean scans | Excellent | Excellent |
| Handwriting | Mediocre | Good |
| Low-quality phone photos | Struggles | Handles well |
| Understands layout semantically (e.g., "this number is the total") | No — needs rules | Yes, natively |
| Structured output with schema | Needs post-processing | Built-in |
| Cost per page | ~$0.0015 (Textract) | ~$0.01 (GPT-4o high detail) |
| Latency | Fast | ~2–5 s |

The rule of thumb: **Tesseract if you have a clean template and volume; LLM if the document is messy, varied, or you want reasoning on top of extraction.** For a personal document tool, the LLM wins. For processing 10 million invoices a day, Textract wins.

### Whisper — audio transcription

OpenAI's `audio.transcriptions` endpoint takes an audio file and returns text.

```python
with open("note.m4a", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",  # optional but helps for short clips
        response_format="verbose_json",  # also returns segments + timestamps
    )
print(result.text)
```

Things worth knowing:

- Max file size: **25 MB**. Longer recordings need to be chunked (ffmpeg) or compressed.
- Formats: mp3, mp4, m4a, wav, webm, mpga, mpeg. Your phone's voice memos work.
- Pricing: **$0.006 per minute** — extremely cheap.
- `response_format="verbose_json"` gives you word-level timing, which matters if you want to sync audio to extracted data later.
- `language=` is optional but improves accuracy on short clips where the model might guess wrong.

For longer recordings (meetings, calls) use `whisper-1` for raw transcription, then feed the result to a chat model for summarization — not the other way around.

### Combining modalities

The most useful pattern in this module: a user photographs a receipt and says, out loud, "this was for the team offsite in Lisbon, split 4 ways." Neither the photo nor the audio alone tells you the whole story. Combined:

```
Photo  → structured extraction (vendor, total, items, date)
Audio  → transcript ("team offsite in Lisbon, split 4 ways")
Fusion → one record with extracted fields + context notes
```

The fusion step is just: transcribe audio, then include the transcript as additional text in the same vision prompt so the model can enrich or correct its extraction. Example prompt:

> "Extract structured data from this receipt. The user also provided a spoken note: '{transcript}'. Use it to fill in context fields (purpose, split, people) that aren't visible in the image."

### Simple storage

No database needed for this module. Append each extracted record to a `docs.json` file:

```json
[
  {
    "id": "a1b2",
    "created_at": "2025-04-17T10:23:00",
    "source_image": "samples/receipt.jpg",
    "note_transcript": "team offsite Lisbon, split 4 ways",
    "extracted": { "vendor": "...", "total": 143.50, ... },
    "cost": { "input_tokens": 1205, "output_tokens": 312, "usd": 0.0065 }
  }
]
```

For `ask`, you load the whole file and stuff it into the context. 100 receipts × 500 tokens = 50k tokens, well within GPT-4o's window. Only worry about retrieval when you hit real scale — and at that point, you already built RAG in Module 04.

---

## Project: Document Intelligence CLI

A single-file CLI you point at an image (and optionally an audio note) that extracts structured data, saves it locally, and lets you query the library in natural language.

### Architecture

```
extract <image>
    │
    └── encode image (base64)
         │
         └── GPT-4o vision + Pydantic schema → ExtractedDocument
              │
              └── print as JSON

analyze <image> --notes <audio>
    │
    ├── transcribe audio via Whisper
    │
    ├── encode image (base64)
    │
    ├── GPT-4o vision + schema + transcript-as-context → ExtractedDocument
    │
    └── append to docs.json

ask <question>
    │
    ├── load docs.json
    │
    ├── build context string (one block per document)
    │
    └── GPT-4o chat completion → answer
```

### Requirements

1. **`extract <image>`** — take a local image path, send to GPT-4o with a Pydantic-based schema, print the structured extraction as JSON
2. **`transcribe <audio>`** — take a local audio path, send to Whisper, print the transcript
3. **`analyze <image> [--notes <audio>]`** — combined pipeline:
   - If `--notes`, transcribe the audio first
   - Send the image (and transcript as additional text context) to GPT-4o
   - Compute cost from token usage
   - Append the record to `docs.json` with a generated `id` and timestamp
4. **`list`** — pretty-print the documents currently in `docs.json` (id, date, vendor/type, total if applicable)
5. **`ask <question>`** — load `docs.json`, stuff all records into the system prompt, ask the model the user's question
6. **Cost tracking** — every vision call records input/output tokens and USD cost in the saved record
7. **Image encoding helper** — detect MIME type from file extension, base64-encode, build the `data:` URL
8. **Graceful failures** — if an image isn't readable, an audio file is too long, or the API errors, print a clear message and exit non-zero

### Starter files

- `app.py` — CLI skeleton with imports, clients, pricing table, Pydantic schemas, storage helpers, and CLI dispatcher wired up. Core functions are stubbed with TODOs.

### Your task

1. Implement `encode_image()` — read bytes, detect MIME type from extension, return `data:image/<mime>;base64,<data>`
2. Implement `extract_document()` — call `client.beta.chat.completions.parse()` with the vision message and the `ExtractedDocument` schema. Return the parsed model plus token usage.
3. Implement `transcribe_audio()` — call the Whisper endpoint, return text and duration.
4. Implement `analyze()` — wire together transcription (optional) + extraction, compute cost, append to `docs.json`.
5. Implement `ask()` — load `docs.json`, build a compact context string, call chat completions, print the answer.
6. Implement `list_docs()` — pretty-print the library.

### Hints

<details>
<summary>Hint for step 1 (encode_image)</summary>

Use `mimetypes.guess_type(path)` to get the MIME type from the extension, then `base64.b64encode(bytes).decode()`. The final string is `f"data:{mime};base64,{b64}"`. GPT-4o accepts `jpeg`, `png`, `webp`, `gif` (non-animated).
</details>

<details>
<summary>Hint for step 2 (extract_document)</summary>

The message content for a vision call is a list of parts, not a plain string:

```python
messages = [
    {"role": "system", "content": "You extract structured data from document images."},
    {"role": "user", "content": [
        {"type": "text", "text": "Extract all fields you can read from this document."},
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ]},
]
```

Use `client.beta.chat.completions.parse(model=..., messages=messages, response_format=ExtractedDocument)`. The parsed result is at `response.choices[0].message.parsed`.
</details>

<details>
<summary>Hint for step 3 (transcribe_audio)</summary>

Open the file in binary mode (`"rb"`) and pass it directly as `file=`. `response_format="verbose_json"` gives you `.duration` along with `.text`, which is handy for logs.
</details>

<details>
<summary>Hint for step 4 (analyze + context fusion)</summary>

When a transcript is present, add a second text part to the user content: `{"type": "text", "text": f"Spoken note from the user: {transcript}"}`. The model will use it to fill in context fields.

For the ID, `uuid.uuid4().hex[:8]` is plenty — this is a local file, not a database primary key.
</details>

<details>
<summary>Hint for step 5 (ask)</summary>

Keep the context string small. For each record, one compact block is enough:

```
[id=abc123 | 2025-04-17 | receipt | vendor=Cafe X | total=14.50 USD]
Notes: team offsite Lisbon, split 4 ways
Full text: ...
```

Put all blocks in the system message, then the user's question as the user message. No tool use, no retrieval — just chat completions.
</details>

---

## Stretch goals

- **PDF support** — accept multi-page PDFs by converting each page to an image (via `pdf2image` / poppler) and extracting per page
- **Handwriting mode** — detect whether the document is handwritten and switch the system prompt to one tuned for handwriting (worth trying both with `high` and `low` detail to see the accuracy/cost curve)
- **Bounding boxes** — ask the model to return pixel coordinates for each extracted field, draw them on the image with Pillow as a debugging overlay
- **Receipt splitter** — given an `analyze` record with a `split` note ("split 4 ways"), produce a per-person breakdown
- **Voice-only mode** — a `voice` subcommand that records from the mic, transcribes, and routes to the right action (extract, ask, list) based on intent

---

## Key questions

- Why does `"detail": "low"` cost ~85 tokens regardless of image size, but `"high"` scales with tiles? What does that tell you about how the model processes images?
- When would you choose Tesseract over GPT-4o for OCR? Give one scenario where each clearly wins.
- You notice your receipts are extracted with 95% accuracy but IDs (passports, driver's licenses) with only 70%. What are the likely causes, and how would you debug?
- If you had to process 50,000 invoices per day, how would you redesign this pipeline? What stays, what changes?
- Structured outputs guarantee the JSON matches your schema, but not that the *values* are correct. How would you validate factual correctness of the extraction?

---

## Resources

- [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [Image token calculator](https://platform.openai.com/docs/guides/vision/calculating-costs)
- [Pydantic docs](https://docs.pydantic.dev/)
