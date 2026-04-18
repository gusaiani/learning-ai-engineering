# Module 12 — System Design for AI

**Goal:** Stop coding for a day. Design a real production AI system on paper — the one you'd pitch as a staff-level engineer on day one of a new job. Make the architecture, trade-offs, numbers, and rollout plan specific enough that someone could build it from your doc.

**Time:** ~1 day

---

## Why this module exists

Everything up to here has been implementation. Modules 03–11 each built one component: embeddings, RAG, agents, evals, streaming, fine-tuning, multi-agent, observability, multimodal. The job of a senior AI engineer is to compose those components into a system that works under real load, real budgets, real users, and real compliance — and to explain *why* it's built that way.

That is what system design interviews test, and what technical leads do every week. The deliverable is not code. It is a document that answers: **what, why, how much, how fast, what fails, and how we'd know**.

---

## What you'll learn

- How to frame an AI system design problem: users, scale, budget, latency targets
- Capacity planning for LLM workloads (tokens/sec, QPS, concurrency, cost per request)
- RAG-at-scale architecture: index tiering, hybrid retrieval, reranking, freshness, multi-tenancy
- Agent system design: tool surface area, loop control, guardrails, cost caps
- Cost / latency / quality — the three-way trade-off and how to reason about it explicitly
- Evals in production: offline suites, online metrics, gold sets, guardrails, shadow traffic
- Rollout strategy: shadow → canary → % ramp → full, with rollback criteria
- Failure modes specific to AI: hallucination, prompt injection, stale context, model deprecation, vendor outage

---

## Concepts

### The shape of an AI system design doc

A good design doc is boring on purpose. It should be scannable in 10 minutes and complete in 45. Use roughly this skeleton — same one used at Anthropic, OpenAI, Google, Meta:

1. **Problem & goals** — one paragraph. What the system does, who uses it, what "good" looks like.
2. **Non-goals** — what this system explicitly does *not* do. Kills scope creep.
3. **Requirements** — functional, plus quantified SLOs (latency p50/p95/p99, availability, accuracy).
4. **Users & traffic** — who calls it, how often, peak vs average, per-tenant skew.
5. **High-level architecture** — one diagram, <10 boxes. Data flow arrows. If it doesn't fit on one page, it's wrong.
6. **Key components** — one short section each: what it does, why it exists, alternatives considered.
7. **Data model** — what you store, where, how it's partitioned (critical for multi-tenant).
8. **Capacity & cost** — back-of-envelope numbers. Tokens/request × QPS × $/token = monthly bill.
9. **Evals & quality** — how you measure the system is working now and regressing later.
10. **Observability** — what you log, what you alert on.
11. **Rollout plan** — shadow, canary, ramp, rollback criteria.
12. **Failure modes & mitigations** — enumerate them. Hallucination, prompt injection, outages, cost runaways.
13. **Open questions** — what you don't know yet. Staff engineers *enumerate* their unknowns; juniors hide them.

The goal is explicitness. If a reviewer has to infer what you meant, you failed.

### Back-of-envelope math for LLM systems

You cannot design an LLM system without estimating numbers. The math is embarrassingly simple, but most people skip it and ship systems that cost 20× what they projected.

**The unit economics of one request:**

```
tokens_in   = system_prompt + retrieved_context + user_input
tokens_out  = response_length
cost        = tokens_in  × $/M_in
            + tokens_out × $/M_out
latency     = TTFT + (tokens_out / tokens_per_second)
```

**Worked example — a support chatbot:**

- System prompt: 1,500 tokens (policies, tone, tools)
- Retrieved context: 5 chunks × 400 tokens = 2,000
- User question: 100 tokens
- Response: 300 tokens
- **Total in:** 3,600 tokens. **Out:** 300 tokens.

At GPT-4o pricing (~$2.50/M in, $10/M out in early 2026):

- Cost/request = 3,600 × $2.50/1M + 300 × $10/1M = **$0.012**
- At 10 QPS sustained: 864,000 req/day × $0.012 = **$10,368/day** = ~$315k/month

That number will change every design decision you make. Maybe you move routine questions to a smaller model. Maybe you cache the system prompt. Maybe you compress context. *All of those now have a dollar value.*

**The four levers you have:**

| Lever | Effect | Trade-off |
|---|---|---|
| Smaller model for easy queries (routing) | -60–80% cost | Added router latency + failure modes |
| Prompt caching (Anthropic / OpenAI) | -50–90% cost on input | Cache eviction; locked prompt structure |
| Context compression / summarization | -30–50% input | Quality loss if over-compressed |
| Shorter outputs (structured vs prose) | -50% output cost + latency | UX: users may want prose |

Every serious design doc includes this table with numbers for *that* system.

### RAG at scale (vs the toy version you built in Module 04)

Module 04 worked with a single PDF corpus and pgvector. At scale the hard problems change:

1. **Index size** — 10M chunks × 1536-dim embeddings = ~60 GB RAM just for vectors. Past ~1M vectors you want an ANN index (HNSW, IVF) not exact search.
2. **Freshness** — documents change. Strategies: periodic re-embed, write-through on edit, CDC from the source-of-truth DB.
3. **Multi-tenancy** — tenant A must not see tenant B's data. Options (worst to best): filter at query time (cheap, risky), per-tenant namespace (good), per-tenant index (best isolation, highest ops cost).
4. **Hybrid retrieval** — dense (embeddings) + sparse (BM25) combined via RRF or a reranker. Dense alone is worse than BM25 on rare-term queries (product codes, error IDs, names).
5. **Reranking** — retrieve 50 → rerank to top 5 with a cross-encoder (Cohere Rerank, Voyage, or a small open model). ~20–40% quality lift for ~50 ms added latency.
6. **Chunking strategy** — no single right answer. Start with recursive 512-token chunks with 50-token overlap; iterate using your eval set.
7. **Query understanding** — rewrite the user's question (HyDE, multi-query, decomposition) before retrieval. Short user queries are underspecified; LLMs are good at expanding them.

A scaled RAG is not "embed everything and search." It is a pipeline: **query rewrite → hybrid retrieval → rerank → context compression → generation → attribution**.

### Multi-tenancy: the most underrated constraint

If your system serves more than one customer, every other design decision bends around tenancy.

- **Data isolation** — how do you guarantee tenant A never sees tenant B's data, even on a bug? Row-level security, per-tenant namespaces, and prompt-level filters. Belt *and* suspenders, because the failure mode is unrecoverable.
- **Noisy neighbors** — one tenant can't hog capacity. Per-tenant rate limits + fair-share scheduling.
- **Per-tenant customization** — different tenants want different system prompts, tools, policies. Design this in from day one; retrofitting it is painful.
- **Cost attribution** — you must know which tenant cost what. Every request carries a `tenant_id` in logs and billing events.
- **Compliance** — some tenants need their data in EU only, or SOC 2 scoped, or HIPAA. Region-pinned indices and per-tenant audit logs.

Single-tenant → multi-tenant is a full rewrite. Multi-tenant from day one is a rounding error.

### Agent system design (beyond "loop until done")

Module 05 built a research agent. That pattern breaks at three points in production:

1. **Cost bombs** — an agent that can call itself recursively will eventually spend $50 on one question. You need a hard token budget and a max-steps cap per invocation, both server-side.
2. **Tool surface area** — the more tools, the worse the agent. Curate aggressively. If two tools overlap, merge them. If a tool is rarely used, gate it behind an intent classifier.
3. **Loop control** — the model decides when to stop. Bad decision = infinite loop. Enforce: (a) max N steps, (b) no-progress detector (same tool call twice with same args = stop), (c) budget exhausted = return partial result with a clear marker.

Agents in prod are orchestrations, not free-running loops. They look more like state machines with LLM-powered transitions.

### Evals in production

You already built an offline eval suite in Module 06. In production you need two more layers:

- **Gold set regression** — a frozen ~100-example suite that runs on every deploy. Blocks release if quality drops.
- **Online metrics** — passive signals from real traffic: thumbs up/down, conversation length, retry rate, human handoff rate, task completion if measurable. Correlate them with model / prompt changes.
- **Guardrails** — real-time classifiers that block or escalate. PII detection, prompt injection, off-topic, policy violations. Cheap models (GPT-4o-mini, Haiku) or fine-tuned classifiers.
- **Shadow traffic** — send real production requests to a candidate (new model, new prompt) in parallel with prod, compare outputs offline. Zero user risk, real distribution.

Rule of thumb: **if you cannot roll a change back in under 10 minutes with clear criteria, you have no rollout strategy.**

### Rollout: shadow → canary → ramp → full

Never flip a flag. Every model change, every prompt change, every retrieval change gets this sequence:

1. **Shadow** (1–3 days) — new pipeline runs alongside prod on 100% of traffic, outputs are logged but not served. Compare quality metrics offline.
2. **Canary** (1% of traffic, 1 day) — new pipeline actually serves 1% of users. Watch error rate, latency, online metrics.
3. **Ramp** — 10%, 25%, 50%, 100% over a week, with auto-pause on regression.
4. **Rollback criteria, written in advance:**
   - Error rate > 2× baseline for 15 min → auto-rollback
   - p95 latency > 1.5× baseline for 15 min → auto-rollback
   - Online quality metric down > 10% day-over-day → manual review, default rollback

Writing rollback criteria *before* rollout is the entire game. If you improvise during an incident, you'll under-rollback and wear the pain for a week.

### AI-specific failure modes

Classic systems fail from load, bugs, and dependencies. AI systems add:

- **Hallucination** — model invents a fact. Mitigation: retrieval + citation, structured outputs, confidence estimation, refusal policy.
- **Prompt injection** — user input (or retrieved content) contains instructions that override yours. Mitigation: input classifiers, output guardrails, never execute tools based on retrieved text without a confirmation step.
- **Stale context** — retrieved data is out of date. Mitigation: freshness SLAs per content type, TTLs, write-through invalidation.
- **Model deprecation** — your vendor retires the model in 6 months. Mitigation: abstract the provider, run your eval suite on candidate replacements quarterly.
- **Vendor outage** — OpenAI is down for 40 minutes. Mitigation: second provider fallback for critical paths, degraded mode (cached answers, human handoff) for everything else.
- **Cost runaway** — a bug causes 10× token usage overnight. Mitigation: per-tenant + global daily spend caps, alerts at 50% / 80% / 100% of budget.

Enumerate these in the doc. Don't hand-wave.

---

## Project: Production Design Doc for a Multi-Tenant AI Support Agent

You are the tech lead. A Series B SaaS company (think Linear, Notion, Figma scale) has hired you to design an AI support agent that:

- Answers customer questions using the company's help center, product docs, and per-tenant workspace data
- Can take actions on the customer's behalf via a small set of tools (create ticket, look up account, reset setting)
- Hands off to a human when it can't resolve confidently
- Is embedded in-product for paying customers and on the marketing site for free visitors
- Must respect per-workspace data boundaries (workspace A never sees workspace B's data)

**Target scale for v1:**
- 5,000 tenants (workspaces)
- 2M end users (customers of those tenants)
- Peak 50 QPS, average 10 QPS
- p95 end-to-end latency ≤ 4s
- Monthly budget cap: **$50,000** in LLM spend
- Availability target: 99.5%

You will not write code. You will write one design doc (`DESIGN_DOC.md`) that any engineer on your team could implement from. A senior reviewer should be able to read it in 45 minutes and either approve it or push back with specific objections.

### Requirements

Your `DESIGN_DOC.md` must include, with real specificity (numbers, component choices, named trade-offs):

1. **Problem, goals, and non-goals** — including quantified SLOs for latency, availability, and quality
2. **High-level architecture diagram** (ASCII or mermaid) with <10 boxes showing request flow
3. **RAG pipeline design**: chunking strategy, embedding model + dimension + storage, retrieval (dense + sparse or not, and why), reranking (or not, and why), context assembly
4. **Multi-tenancy strategy**: data isolation, per-tenant namespaces vs shared index, noisy-neighbor controls, cost attribution
5. **Agent design**: tool list (name + purpose + who can call it), loop control (max steps, budget, no-progress detection), human-handoff trigger
6. **Model choices & routing**: which model handles which request types, why, with a routing diagram
7. **Capacity & cost math**: show your work. Tokens per request type × QPS × pricing → monthly $. Demonstrate you fit the $50k cap, or explain what breaks if you don't.
8. **Evals**: offline gold set (size, composition), online metrics you'll track, guardrails (what they check, what model runs them), shadow testing approach
9. **Observability**: trace schema (what fields on every request), dashboards (5 key charts), alerts (5 must-have pages)
10. **Rollout plan** for v1 launch AND for future model/prompt changes, with *written* rollback criteria (specific thresholds, not "if things look bad")
11. **Failure modes table**: at least 8 rows. Each row: failure, likelihood, impact, mitigation, detection
12. **Open questions**: at least 5. Things you genuinely don't know and would need to investigate before or after launch
13. **Phased delivery**: v1 (week 1–6), v1.5 (month 3), v2 (month 6). What's in, what's out, why.

### Starter files

- `DESIGN_DOC.md` — skeleton with every section heading, guiding prompts inside each section as HTML comments (`<!-- ... -->`), and a few reference tables pre-filled to match the prior-module style. You fill in all the substance.

### Your task

Work through the doc top-to-bottom. Resist the urge to jump to architecture first — the numbers drive the architecture, not the other way around. In order:

1. **Lock the problem & SLOs.** One paragraph. Non-goals. Quantified latency / availability / quality targets.
2. **Do the capacity & cost math first.** How many requests per day? How many tokens per request type? What does that cost at today's prices? This determines whether you can afford GPT-4o everywhere, need a router, or need caching as a hard requirement.
3. **Sketch the architecture.** One diagram, <10 boxes. Request enters → where does it go? When does it hit the LLM? When does it return?
4. **Design the RAG pipeline.** Be specific about embedding model, index type, chunking, retrieval strategy, reranking.
5. **Design for multi-tenancy explicitly.** Do not hand-wave "we filter by tenant_id." Write out the guarantee and the belt-and-suspenders.
6. **Design the agent loop.** Tools, caps, handoff.
7. **Design evals and rollout.** Be specific about rollback thresholds.
8. **Enumerate failure modes.** Minimum 8 rows. Don't skip boring ones.
9. **Write the open questions.** If you have fewer than 5, you're not being honest — every real system has unknowns.
10. **Read it top-to-bottom.** Could someone build from this? Is there a decision you've hidden behind a vague word? Fix those.

### Hints

<details>
<summary>Hint for step 2 (cost math)</summary>

Break requests into 3–4 classes with different shapes:

| Class | % of traffic | Tokens in | Tokens out | Notes |
|---|---|---|---|---|
| FAQ (short answer from help center) | 60% | 2,500 | 150 | Could use a cheaper model |
| Account action (agent + 1–2 tool calls) | 25% | 4,500 | 400 | Needs stronger model |
| Complex (multi-step agent) | 10% | 8,000 | 600 | Stronger model, budget capped |
| Handoff (summarize to human) | 5% | 3,000 | 200 | Any model |

Average tokens_in × $/M + tokens_out × $/M, weighted by % of traffic. Multiply by requests/day × 30. Compare to your $50k cap. If you're over, you now know *exactly* which lever to pull.
</details>

<details>
<summary>Hint for step 3 (architecture diagram)</summary>

Keep it under 10 boxes. A reasonable set:

```
Client → API Gateway → Router (intent classifier)
                         ├──→ Small-model FAQ path → Response
                         └──→ Agent path → Tools / RAG retriever → Large model → Response
                                                ↑
                                          Vector store + BM25 index
Shared across all paths: Guardrail layer · Trace logger · Cost tracker
```

Don't try to show everything. The goal is *request flow*, not a complete infra diagram.
</details>

<details>
<summary>Hint for step 5 (multi-tenancy)</summary>

Belt + suspenders means *two* independent mechanisms. Example:

1. Every vector in the index has a `tenant_id` attribute, queried with a mandatory filter
2. The retrieval service is a dedicated per-tenant namespace (Pinecone namespaces, Qdrant collections, or separate pgvector schemas)

If the query-time filter is ever omitted due to a bug, the namespace isolation still prevents cross-tenant bleed. This is the kind of design detail that separates "I've thought about it" from "I've actually built multi-tenant before."
</details>

<details>
<summary>Hint for step 7 (rollout)</summary>

Good rollback criteria are numeric, not vibes:

- **Bad:** "roll back if quality drops"
- **Good:** "auto-rollback if the thumbs-down rate over any 30-minute window exceeds 2× the prior 7-day baseline"

Write three or four specific triggers. For each, state: metric, threshold, window, action (auto vs manual), who's paged.
</details>

<details>
<summary>Hint for step 8 (failure modes)</summary>

Minimum 8 rows, and *actually* use the full table:

| Failure | Likelihood | Impact | Mitigation | Detection |
|---|---|---|---|---|
| Vendor outage (primary LLM) | Monthly | High | Failover to secondary provider; degraded mode | Provider status + synthetic checks every 60 s |
| Prompt injection via retrieved doc | Weekly | Medium | Untrusted-content delimiter; output guardrail classifier | Guardrail classifier logs |
| ... | ... | ... | ... | ... |

Make sure you cover: hallucination, prompt injection, stale context, cost runaway, cross-tenant leak, model deprecation, vendor outage, tool failure, partial retrieval failure. That's already 9.
</details>

---

## Stretch goals

- **Add a second diagram** showing the eval + rollout infrastructure as its own subsystem (shadow traffic tap, offline eval runner, online metrics pipeline, rollout controller)
- **Add a "what changes at 10× scale" section** — you hit 500 QPS and 50,000 tenants. Which design decisions break? (Answer: probably the embedding-every-doc-synchronously ingestion pipeline, and maybe the shared rerank model.)
- **Write a shorter "1-pager"** — force yourself to compress the whole system onto a single page (problem, architecture, numbers, risks). This is what you'd actually send to a VP.
- **Red-team your own doc** — write a half-page adversarial review from the perspective of a skeptical staff engineer. What would they push back on? Fix the doc until those pushbacks are already answered in it.
- **Do a second design** for a different system: an AI coding assistant like Cursor, an AI legal research tool, or an AI voice agent. Different constraints produce different architectures — that's the lesson.

---

## Key questions

- Why would you design this as an agent (tool-calling loop) instead of a fixed RAG pipeline with a few if-statements? What does the agent buy you that's worth its cost and failure modes?
- Your cost math shows you're $20k/month over the $50k budget. Name the three levers you'd pull, in order, and quantify the expected savings from each.
- A tenant reports that one of their customers received an answer that referenced *another tenant's* internal documentation. Walk through: how could this happen given your design, how would you detect it, and how would you prevent it in a way a code reviewer could enforce?
- Your p95 latency is 6 seconds but the target is 4. Where does the time go, and which single change is most likely to bring it under 4 without hurting quality?
- You need to upgrade the primary model from GPT-4o to a hypothetical "GPT-5." Walk through the rollout — what runs first, what metrics gate each step, what triggers an auto-rollback, who's on-call?
- A month after launch, a guardrail model that blocks prompt-injection attempts is catching 0.3% of requests. How do you know whether that's correct, too high, or too low?

---

## Resources

- [Google's "Design Docs at Google" (Industrial Empathy)](https://www.industrialempathy.com/posts/design-docs-at-google/)
- [Evan Miller — "How Not to Run A/B Tests"](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) (relevant for rollout thresholds)
- [Chip Huyen — "Designing ML Systems"](https://huyenchip.com/ml-interviews-book/) — ML systems lens, mostly pre-LLM but frameworks carry over
- [Hamel Husain — "Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/)
- [Eugene Yan — "Patterns for Building LLM-based Systems & Products"](https://eugeneyan.com/writing/llm-patterns/)
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents)
- [OpenAI — "A practical guide to building agents"](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [Pinecone — Multi-tenancy in vector DBs](https://www.pinecone.io/learn/multitenancy/)
