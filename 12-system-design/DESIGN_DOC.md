# Design Doc: Multi-Tenant AI Support Agent

**Author:** Gustavo Saiani
**Status:** Draft
**Last updated:** 2026-04-17
**Reviewers:** Jensen Huang, Dario Amodei, Warren Buffett

---

## 1. Problem & Goals

The support team at Sponda handles an average of 10 QPS from 2M end users across 5,000 customer workspaces. Today, all traffic routes to humans; resolution time averages 45 minutes and throughput caps out at ~20 concurrent conversations. This proposal design an AI-powered agent that answers 80% of inbound questions without human involvement, reducing resolution time to ≤ 4 seconds p95, and freeing support staff to handle complex escalations. The agent must respect strict data isolation (workspace A never sees workspace B's docs or settings) and operate within a $50k/month LLM budget.

### Goals

- Resolve 80% of inbound support questions without human escalation (measured by auto-resolve rate on gold set)
- Achieve p95 end-to-end latency ≤ 4 seconds per request
- Ensure zero cross-tenant data leakage across 100% of production requests
- Stay within $50k/month LLM budget even at peak QPS
- Maintain availability ≥ 99.5% monthly

### Non-goals

- Build a phone/voice support agent (text-only for v1)
- Fine-tune custom models (use off-the-shelf models only)
- Proactive outreach or notifications
- Support for non-English languages in v1
- Real-time document sync (async refresh acceptable)

### Quantified SLOs

| Metric                             | Target          |
| ---------------------------------- | --------------- |
| p50 end-to-end latency             | ≤ 1.5s          |
| p95 end-to-end latency             | ≤ 4s            |
| p99 end-to-end latency             | ≤ 8s            |
| Availability                       | ≥ 99.5% monthly |
| Answer-correctness (offline gold)  | ≥ 80%           |
| Human-handoff rate (healthy range) | 10-20%          |
| Monthly LLM spend                  | ≤ $50,000       |

---

## 2. Users & Traffic

<!--
- Who calls this system? (end users? tenant admins? internal tools?)
- What's the traffic shape? Peak vs average QPS, burstiness, per-tenant skew.
- Authentication: how do we identify the tenant and the user?
-->

| Dimension            | Value                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------- |
| Tenants (workspaces) | 5000                                                                                   |
| End users            | 2,000,000                                                                              |
| Avg QPS              | 10                                                                                     |
| Peak QPS             | 50                                                                                     |
| Per-tenant skew      | Top 1% of tenants drive 35% of traffic; top 10% drive 70%                              |
| Auth                 | Workspace JWT (workspace_id + user_id) for in-product; anon session for marketing site |

---

## 3. High-Level Architecture

```
  Client → API Gateway → Router (intent classifier)
                           ├──→ FAQ path (small model) → Response
                           └──→ Agent path → RAG retriever → Large model → Response
                                              ↓
                                      Vector store + BM25
  Shared: Guardrail layer · Trace logger · Cost tracker
```

### Request walkthrough

1. Client sends request to API Gateway with workspace JWT + question
2. Gateway authenticates JWT, rate-limits per workspace, logs request
3. Router classifier reads question, assigns 90% confidence to "FAQ" class
4. FAQ path: retriever fetches 5 chunks from help center (BM25 + dense retrieval)
5. Guardrail layer checks retrieved content for PII/injection (quick pass)
6. Small model (gpt-4o-mini) generates answer from context in 200 tokens
7. Output guardrail checks answer doesn't leak data from other workspaces
8. Trace logger records: workspace_id, tokens_in, tokens_out, cost, latency
9. Response returned to client (~1.2s p50 latency, $0.0015 cost)

---

## 4. Key Components

### 4.1 API Gateway / Ingress

Handles HTTP routing, authentication (JWT validation), and rate limiting. Every request starts here. We enforce per-workspace rate limits (100 req/s per workspace) and per-user limits (10/req/s) to prevent noisy neighbors. Could have skipped this and rate-limited at the service layer, but centralizing at the gateway gives us fast rejection before expensive processing, and is a standard production pattern.

### 4.2 Router / Intent Classifier

Routes requests by intent using gpt-4o-mini in <200ms. Four classes: FAQ (help center lookup), Account action (simple tool use), Complex (multi-step reasoning), Handoff (escalation summary). Low-confidence classifications default to Agent path for safety.

### 4.3 RAG Pipeline

See section 5 for details.

### 4.4 Agent Loop

See section 7 for details.

### 4.5 Guardrail Layer

Runs pre-request and post-response classifiers to block malicious or policy-violating content. Pre-request checks: PII in user input, prompt injection, off-topic abuse. Post-response checks: PII leakage, cross-tenant data references. All checks use gpt-4o-mini for speed and cost.

### 4.6 Trace Logger / Cost Tracker

Emits a structured trace on every request with: request_id, tenant_id, user_id, route_class, tokens_in, tokens_out, cost_usd, latency_ms, tool_calls, guardrail_triggers, handoff status. Traces feed billing (per-tenant cost attribution), observability dashboards, and rollout monitoring.

---

## 5. RAG Pipeline

The RAG pipeline serves two very different knowledge classes: shared product knowledge (help center, API docs, policy docs) and tenant-private knowledge (workspace settings, plan details, prior tickets, tenant-authored docs). We index them separately, attach strict metadata to every chunk (`workspace_id`, `document_type`, `source_id`, `updated_at`, and access control metadata), and only retrieve from the scopes authorized for the current request. Retrieval uses a hybrid approach: lexical search for exact product terms and error strings, dense search for semantic matching, and a lightweight reranker before context is passed to the generator.

The design goal is to maximize answer quality without blowing the latency or isolation budget. In practice that means: asynchronous ingestion rather than real-time sync, structure-aware chunking instead of naive fixed windows, small embeddings for cost efficiency, top-k pruning before generation, and explicit citations back to source chunks so answers remain auditable.

### 5.1 Corpus

The corpus has two buckets: shared public knowledge and tenant-private knowledge. Shared public knowledge includes help center articles, product documentation, API docs, pricing and plan docs, troubleshooting guides, and policy pages. Tenant-private knowledge includes workspace settings, enabled integrations, plan entitlements, tenant-authored internal docs, past support tickets, and resolved conversation summaries. We do not index raw chat transcripts by default in v1 because they are noisy, privacy-sensitive, and expensive to keep fresh.

Only sources with stable support value are indexed. We exclude secrets, API keys, payment instrument data, one-time tokens, and ephemeral event streams. Public docs are re-crawled on a scheduled basis, while tenant-private sources are updated asynchronously on write through an ingestion queue so the index usually reflects changes within a few minutes rather than requiring real-time sync.

### 5.2 Chunking

We use structure-aware recursive chunking rather than fixed-size splitting. For public docs, we first split on natural boundaries such as article sections, headings, lists, and table blocks, then apply token-based chunking within those sections only when needed. For tenant-private records such as tickets or settings pages, we chunk by logical object boundaries first so one chunk corresponds to one coherent support issue or configuration state.

Target chunk size is 300 to 500 tokens with 50 to 75 tokens of overlap. This range is large enough to preserve procedural context and product terminology, but small enough to avoid embedding entire articles, diluting retrieval precision, or wasting prompt budget on near-duplicate context.

Overlap is only applied when a section must be split across multiple chunks; we do not overlap unrelated sections because that creates duplicate embeddings and hurts ranking quality. We also store heading path and source metadata with every chunk so retrieval can return both the passage and its surrounding context, such as `Billing > Invoices > Failed payment`.

### 5.3 Embedding

| Choice           | Value                                                                              | Why                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Model            | text-embedding-3-small                                                             | Good retrieval quality at low cost; cheap enough to re-embed frequently as docs change                          |
| Dimensions       | 1536                                                                               | Strong quality/latency tradeoff without the storage cost of larger embedding footprints                         |
| Storage          | Postgres + pgvector for dense search, plus BM25 index in OpenSearch                | Keeps the architecture simple and transactional for tenant metadata while still supporting hybrid retrieval     |
| Re-embed cadence | On document write for tenant-private data; nightly backfill for shared public docs | Keeps private data fresh enough for support use cases without requiring expensive real-time recrawls everywhere |

### 5.4 Retrieval

We use hybrid retrieval: dense vector search for semantic matching and BM25 for exact keyword matching. This matters in support because users often paste exact error strings, feature names, invoice terms, or configuration labels that lexical search handles better than dense embeddings alone.

We fetch the top 20 candidates from dense search and the top 20 from BM25, merge them by normalized score, and deduplicate by source chunk ID. We then send the top 8 merged candidates through a lightweight reranker and keep the best 4 to 6 chunks for final context assembly.

The reranker can be a small cross-encoder or an LLM-based relevance check, but it must add no more than about 100 to 150 ms p95 latency. This extra stage is worth it because first-pass retrieval often returns partially relevant chunks, and reranking improves precision before the generator consumes scarce prompt budget.

### 5.5 Query understanding

<!--
Do you rewrite / expand / decompose the user query before retrieval? How?
-->

### 5.6 Context assembly

<!--
- Token budget for retrieved context
- How you handle "too many good chunks" (compression? just truncate?)
- Citation format returned to the generator
-->

---

## 6. Multi-Tenancy

<!--
This is the section most design docs fail on. Be paranoid here.
-->

### 6.1 Data isolation

<!--
How is workspace A's data guaranteed not to reach workspace B?
Name at least TWO independent mechanisms (belt + suspenders).
-->

### 6.2 Per-tenant customization

<!--
- Custom system prompt per tenant?
- Per-tenant tool config / allowlist?
- Per-tenant model overrides (e.g. enterprise tenants on a stronger model)?
-->

### 6.3 Noisy-neighbor controls

<!--
- Per-tenant rate limits (requests/min, tokens/day)
- Per-tenant spend caps
- Fair-share queue? Priority tiers?
-->

### 6.4 Cost attribution

<!-- Every request carries tenant_id in the trace. Billing event schema. -->

---

## 7. Agent Design

### 7.1 Tools

| Tool                    | Purpose | Who can call | Side effects | Requires confirmation? |
| ----------------------- | ------- | ------------ | ------------ | ---------------------- |
| <!-- create_ticket -->  |         |              | Writes       | Yes                    |
| <!-- lookup_account --> |         |              | Read-only    | No                     |
| <!-- reset_setting -->  |         |              | Writes       | Yes                    |
| <!-- ... -->            |         |              |              |                        |

### 7.2 Loop control

| Guard                     | Value                                     |
| ------------------------- | ----------------------------------------- |
| Max steps                 | <!-- e.g. 8 -->                           |
| Max tokens per invocation | <!-- e.g. 30,000 -->                      |
| Max wall-clock            | <!-- e.g. 30s -->                         |
| No-progress detection     | <!-- e.g. same tool+args twice = halt --> |
| Budget-exhausted behavior | <!-- return partial + clear marker -->    |

### 7.3 Human handoff

<!--
When does the agent give up and escalate?
- Low confidence on answer?
- User explicitly asks?
- Tool failure?
- Budget exhausted?
What does "handoff" look like concretely — ticket? chat transfer? email? with what summary?
-->

---

## 8. Model Choices & Routing

<!--
Don't use one model for everything. Route by request class.
-->

| Request class             | % traffic    | Model                              | Why                                               |
| ------------------------- | ------------ | ---------------------------------- | ------------------------------------------------- |
| FAQ (short, doc-grounded) | <!-- 60% --> | <!-- gpt-4o-mini -->               | <!-- cheap, accurate enough for simple lookup --> |
| Account action (agent)    | <!-- 25% --> | <!-- gpt-4o -->                    | <!-- tool use reliability -->                     |
| Complex multi-step        | <!-- 10% --> | <!-- gpt-4o -->                    | <!-- reasoning -->                                |
| Handoff summary           | <!-- 5% -->  | <!-- gpt-4o-mini -->               | <!-- summarization is easy -->                    |
| Guardrails                | 100%         | <!-- gpt-4o-mini or classifier --> | <!-- cheap, fast -->                              |

### Router

<!--
- What classifies incoming requests into classes?
- What model / heuristic?
- Added latency budget (should be < 200ms)
- What happens on low-confidence classification?
-->

---

## 9. Capacity & Cost Math

<!--
SHOW YOUR WORK. This is the single most important section.
-->

### 9.1 Request mix

<!-- fill in the table from section 8 with token counts -->

| Class                 | %    | tokens_in | tokens_out | Model | $/req |
| --------------------- | ---- | --------- | ---------- | ----- | ----- |
| FAQ                   |      |           |            |       |       |
| Account action        |      |           |            |       |       |
| Complex               |      |           |            |       |       |
| Handoff               |      |           |            |       |       |
| Guardrail (every req) | 100% |           |            |       |       |

### 9.2 Monthly cost

```
avg $/req   = Σ (class_share × class_$/req)   = $ ____
requests/mo = avg QPS × 86,400 × 30           = ____
LLM cost/mo = avg $/req × requests/mo         = $ ____
+ embeddings + reranker + guardrails          = $ ____
Total                                          = $ ____
```

<!-- If this exceeds $50k, name the lever(s) you pull and re-run the math. -->

### 9.3 Infrastructure cost (non-LLM)

<!-- Vector DB, Postgres, cache, compute, egress. Rough numbers. -->

---

## 10. Evals & Quality

### 10.1 Offline gold set

<!--
- Size
- Composition (what request classes, which tenants represented, failure-mode examples included?)
- How often it runs (every PR? nightly? pre-release?)
- Pass threshold
-->

### 10.2 Online metrics

<!--
What passive signals do you log per request? Which correlate with quality?
-->

| Metric                | Source | Use                           |
| --------------------- | ------ | ----------------------------- |
| Thumbs up/down        | User   | Primary quality signal        |
| Conversation length   | System | Proxy for unresolved          |
| Retry / rephrase rate | System | Proxy for wrong-answer        |
| Human-handoff rate    | System | Primary "agent failed" signal |
| <!-- ... -->          |        |                               |

### 10.3 Guardrails

| Guardrail                | Checks | Model | Action on trigger |
| ------------------------ | ------ | ----- | ----------------- |
| PII leak (output)        |        |       |                   |
| Prompt injection (input) |        |       |                   |
| Off-topic / abuse        |        |       |                   |
| Cross-tenant reference   |        |       |                   |

### 10.4 Shadow traffic

<!-- How do you test a candidate change on real traffic without user risk? -->

---

## 11. Observability

### 11.1 Trace schema

<!-- Every request emits a trace with these fields: -->

```
request_id, tenant_id, user_id, route_class, model,
tokens_in, tokens_out, cost_usd, latency_ms,
retrieval_top_k, rerank_latency_ms,
tool_calls[], guardrail_triggers[], handoff: bool,
online_feedback: {thumb, retry, resolved}
```

### 11.2 Dashboards (pick 5)

<!--
1.
2.
3.
4.
5.
-->

### 11.3 Alerts (pick 5 must-haves)

| Alert | Trigger | Severity | Page? |
| ----- | ------- | -------- | ----- |
|       |         |          |       |
|       |         |          |       |
|       |         |          |       |
|       |         |          |       |
|       |         |          |       |

---

## 12. Rollout Plan

### 12.1 v1 launch

<!--
1. Internal dogfood (1–2 weeks)
2. Shadow traffic on 100% (1 week)
3. Canary on 1% of tenants (3 days)
4. Ramp: 10% → 25% → 50% → 100% over a week
5. GA
-->

### 12.2 Rollback criteria (written in advance)

| Trigger                    | Threshold                 | Window  | Action                        |
| -------------------------- | ------------------------- | ------- | ----------------------------- |
| Error rate spike           | <!-- > 2× 7d baseline --> | 15 min  | Auto-rollback                 |
| p95 latency spike          |                           | 15 min  | Auto-rollback                 |
| Thumbs-down rate           |                           | 30 min  | Page on-call, manual rollback |
| Cost burn rate             | <!-- > 1.5× expected -->  | 1 hr    | Page, manual                  |
| Cross-tenant leak detected | any                       | instant | Auto-rollback + incident      |

### 12.3 Ongoing changes

<!-- Every model / prompt / retrieval change follows: shadow → canary → ramp. Document who owns each gate. -->

---

## 13. Failure Modes

<!-- Minimum 8 rows. Include the boring ones. -->

| Failure                                  | Likelihood | Impact | Mitigation | Detection |
| ---------------------------------------- | ---------- | ------ | ---------- | --------- |
| Hallucinated answer                      |            |        |            |           |
| Prompt injection via retrieved content   |            |        |            |           |
| Cross-tenant data leak                   |            |        |            |           |
| Vendor (primary LLM) outage              |            |        |            |           |
| Model deprecation                        |            |        |            |           |
| Stale retrieval (doc updated, index not) |            |        |            |           |
| Cost runaway                             |            |        |            |           |
| Tool call failure                        |            |        |            |           |
| <!-- add more -->                        |            |        |            |           |

---

## 14. Open Questions

<!--
At least 5. Things you genuinely don't know.
-->

1. <!-- e.g. Is a cross-encoder reranker worth the latency at our scale, or is top-k dense+BM25 enough? -->
2. <!--  -->
3. <!--  -->
4. <!--  -->
5. <!--  -->

---

## 15. Phased Delivery

| Phase | Window    | Scope | Out of scope |
| ----- | --------- | ----- | ------------ |
| v1    | Weeks 1–6 |       |              |
| v1.5  | Month 3   |       |              |
| v2    | Month 6   |       |              |

---

## 16. Appendix

### 16.1 Alternatives considered

<!-- For the biggest 2–3 decisions, what else did you consider and why did you reject it? -->

### 16.2 Glossary

<!-- Any terms a reviewer outside your team might not know. -->
