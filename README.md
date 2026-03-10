# AI Engineering Course

A project-based course to go from zero to $200k+ AI Engineer, working remotely from Brazil.

**Goal:** Build real things every module. No toy examples. Everything goes in a portfolio.

---

## Roadmap

| # | Module | Key Skill | Project | Status |
|---|--------|-----------|---------|--------|
| 01 | [LLM API Fundamentals](./01-llm-api-fundamentals/) | First real API calls, system prompts, structured outputs | CLI chat app | ✅ |
| 02 | [Prompt Engineering](./02-prompt-engineering/) | Chain-of-thought, few-shot, formatting, reliability | Prompt testing harness | ✅ |
| 03 | [Embeddings & Semantic Search](./03-embeddings-semantic-search/) | Vectors, cosine similarity, pgvector | Semantic search over docs | |
| 04 | [RAG — Retrieval Augmented Generation](./04-rag/) | Chunking, retrieval, reranking | RAG Q&A over a PDF corpus | |
| 05 | [AI Agents & Tool Use](./05-agents-tool-use/) | Function calling, tool loops, ReAct | Research agent | |
| 06 | [Evals — Measuring LLM Quality](./06-evals/) | LLM-as-judge, regression testing, benchmarks | Eval suite for your RAG | |
| 07 | [Streaming & Production Patterns](./07-streaming-production/) | Streaming, async, error handling, caching | Production-ready API | |
| 08 | [Fine-tuning](./08-fine-tuning/) | LoRA, QLoRA, datasets, PEFT | Fine-tune a model on custom data | |
| 09 | [Multi-agent Systems](./09-multi-agent/) | Orchestration, parallelism, handoffs | Multi-agent research pipeline | |
| 10 | [Observability & LLMOps](./10-llmops/) | Tracing, cost tracking, latency, monitoring | Instrumented production app | |
| 11 | [Multimodal — Vision & Audio](./11-multimodal/) | Image inputs, audio transcription, OCR | Document intelligence app | |
| 12 | [System Design for AI](./12-system-design/) | Architecture, trade-offs, scalability | Design doc for a real system | |
| 13 | [Capstone](./13-capstone/) | Everything | Ship a complete AI product | |

---

## How to use this course

1. **Do modules in order.** Each builds on the previous.
2. **Build the project before reading the solution.** Struggle is part of it.
3. **Commit everything.** This repo is your portfolio.
4. **Ship each project.** Deploy it, write a short post, put it on LinkedIn.

---

## Stack

- **Language:** Python 3.12+
- **Primary API:** Anthropic (Claude) — also covers OpenAI where needed
- **Vector DB:** pgvector (local via Docker), Pinecone for production
- **Framework:** Mostly from scratch, then LangChain/LangGraph where it earns its place
- **Deploy:** FastAPI + Railway/Render (free tier friendly)
- **Eval:** Braintrust or custom harness
- **Observability:** Langfuse (self-hosted) or LangSmith

---

## Setup

```bash
# Python 3.12+
python -m venv .venv
source .venv/bin/activate

pip install anthropic openai python-dotenv fastapi uvicorn
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Career strategy (Brazil → $200k remote)

The target is US-based companies that hire internationally. That means:

- **GitHub portfolio matters more than a degree.** Every project here is a public artifact.
- **Target companies:** AI-native startups (Series A–C), not FAANG. They pay well and hire globally.
- **Job boards:** Wellfound (AngelList), Otta, Greenhouse, direct outreach on LinkedIn.
- **Rate:** As a contractor (PJ), aim for $80–120/hr. As a full-time remote employee, $150–200k base + equity.
- **Stack to emphasize on resume:** RAG, agents, evals, LangChain/LangGraph, Claude/OpenAI APIs, production deployment.

After Module 6, you are already hirable for a mid-level role. After Module 10, senior.

---

