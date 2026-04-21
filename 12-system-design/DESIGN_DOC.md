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

We rewrite queries selectively rather than on every request. Support queries fall into two rough buckets: precise queries that already retrieve well (pasted error strings, specific feature names, invoice IDs) and vague or context-dependent queries where the raw user text is a poor retrieval key ("it's still broken", "how do I fix this", "what about the other plan"). Rewriting every query would add a full LLM round-trip to the hot path for no benefit on the precise bucket, which 5.4's hybrid retrieval already handles well via BM25. So a lightweight gate decides per request: skip rewriting when the query contains strong lexical anchors (error codes, quoted strings, known product nouns) and invoke the rewriter otherwise.

When rewriting is triggered, we do three things in a single small-model call, capped at roughly 150 to 200 ms p95. First, **context resolution**: resolve pronouns and references against the last two or three turns of conversation so "that error" or "the other plan" becomes a self-contained query. Second, **expansion**: generate one or two paraphrases that add likely product terminology the user did not use, which improves recall on the dense side without polluting BM25. Third, **decomposition**: if the query contains multiple independent questions ("how do I enable SSO and what does it cost on the Team plan"), split it into separate retrieval calls whose results are merged before generation. We deliberately do not use HyDE-style hypothetical-answer generation in v1; it is expensive, and our hybrid retriever plus reranker already closes most of the gap HyDE is designed to close.

The rewriter runs on a small, fast model (same tier as the intent classifier) with a strict timeout. If it times out or returns malformed output, we fall back to the raw user query rather than blocking the request, and log the failure for offline analysis. Every rewritten query is stored on the trace alongside the original so evals, debugging, and regression tests can compare retrieval quality with and without rewriting per query class.

### 5.6 Context assembly

We allocate roughly 2,500 tokens of the generator's input budget to retrieved chunks, out of a total input ceiling of about 6,000 tokens per request. The remaining ~3,500 tokens cover the system prompt, the last two or three turns of conversation history, tool schemas on the agent path, and the user's current question, with a safety margin. This sits well inside the 128k context window of the models in section 8, but context-window size is not the constraint: input tokens drive both cost and latency, and generator quality degrades once context is padded with weakly relevant material ("lost in the middle"), so the budget is set for the answer quality sweet spot, not the model's technical maximum.

The 2,500-token figure is derived from 5.4, which caps us at 4 to 6 reranked chunks, and 5.2, which targets 300 to 500 tokens per chunk - giving a natural range of 1,600 to 2,400 tokens of dense retrieval content with a small cushion for heading paths and source metadata we inject per chunk. The budget is a soft cap enforced at assembly time rather than a hard retriever limit: reranking has already pruned to a small candidate set, so assembly rarely needs to drop chunks, and when it does we apply the rules in step 2 rather than silently truncating mid-chunk.

When reranked chunks exceed the 2,500-token budget, we drop the lowest-scoring chunks until we fit, rather than compressing or truncating. Reranking has already produced a precision-ordered list, so the marginal chunk is by construction the least useful one to keep; dropping it costs the least expected answer quality. Truncating mid-chunk is off the table because it orphans sentences from their heading path and source metadata, which the generator relies on for citation accuracy (step 3). LLM-based compression is tempting but adds a synchronous model call on the hot path — 200 to 500 ms and a real dollr cost per request — for a gain we have not measured, and it introduces a lossy transform before the generator even sees the evidence, making retrieval bugs harder to diagnose from traces.

---

## 6. Multi-Tenancy

### 6.1 Data isolation

Tenant isolation is the failure mode we treat as catastrophic: one leak of workspace A's data into workspace B's response is worse than hours of downtime. So we enforce isolation with two independent mechanisms whose failure modes do not overlap, rather than trusting any single layer.

**Mechanism 1 — Postgres row-level security (RLS).** Every tenant-scoped table carries a non-null `workspace_id` column, and RLS policies restrict `SELECT`, `INSERT`, `UPDATE`, and `DELETE` to rows where `worskspace_id = current_setting('app.workspace_id')::uuid`. The session variable is set from the authenticated request context at the start of every transaction and cannot be forged by application code further down the stack. This catches the class of bugs where a developer writes a query that forgets the `WHERE workspace_id = ?` clause — the database itself refuses to return cross-tenant rows, so an application bug degrades to "no results" rather than "wrong tenant's results".

**Mechanism 2 — retrieval-time scope filter in the RAG path.** The vector store and BM25 index both store `workspace_id` as required metadata on every chunk, and the retrieval service rejects any query that does not supply an explicit `workspace_id` filter resolved from the request's auth context, not from user input. Shared public knowledge is indexed under a reserved `workspace_id = "__shared__"` scope that every tenant can read but no tenant can write. This catches the class of failures RLS cannot catch: a misconfigured RLS policy, a direct connection that bypasses the session variable, or a future migration to a vector store that has no equivalent of RLS. Retrieval still refuses to return another tenant's chunks because the filter is applied in application code above the store.

The two mechanisms fail independently. RLS protects against forgotten `WHERE` clauses but trusts that the session variable is set correctly; the retrieval filter protects against misconfigured RLS or a non-Postgres datastore but trusts that the application resolves `workspace_id` from auth rather than from request bodies. A single bug has to defeat both layers to leak data, and the two layers are owned by different parts of the codebase (DB migrations vs. retrieval service), so a single careless commit is unlikely to weaken both at once.

We also enforce a tertiary check at the trace layer: every outbound response is tagged with the `workspace_id` it was generated for, and a post-response guardrail rejects responses that cite chunks whose `workspace_id` does not match. This is not counted as one of the two primary mechanisms because it runs after generation and is a detector rather than a preventer, but it closes the loop by making cross-tenant leaks observable in production rather than silent.e

### 6.2 Per-tenant customization

Per-tenant customization is limited by design. Every knob we expose is a knob an attacker or a misconfigured admin can turn, and the isolation guarantees from §6.1 are only as strong as the config layer that drives them. We allow three customization surfaces and explicitly forbid the rest in v1.

**System prompt overrides.** Tenants on Business and Enterprise plans can append a short "voice and policy" block to the base system prompt — brand name, tone guidance, escalation phrasing, off-limits topics specific to their product. The base prompt is immutable and always wins on conflicts: tenant overrides cannot disable guardrails, cannot grant tools, cannot change the citation requirement, and are capped at roughly 500 tokens to bound cost and prompt-injection surface. Overrides are versioned and reviewed in the admin UI diff view before they go live, and every trace records the prompt version so regressions tie back to a specific change.

**Tool allowlist.** The full tool catalog from §7.1 is opt-in per tenant. A fresh workspace starts with the read-only tools enabled (`lookup_account`, `search_kb`) and write tools disabled. Admins enable write tools (`create_ticket`, `reset_setting`) explicitly, and each write tool can be individually scoped — e.g. `reset_setting` enabled but only for a named subset of settings keys. The allowlist is enforced at the agent-loop layer: a tool call for a disabled tool is rejected before it reaches the tool implementation, and the rejection is surfaced to the user as "this workspace has not enabled that action" rather than a silent failure.

**Model tier.** Enterprise tenants can opt into a stronger generator model for the Account-action and Complex-multi-step request classes from §8 — typically the flagship model instead of the default mid-tier. The router still classifies requests the same way; only the model binding for the two high-reasoning classes changes. FAQ and guardrail classes are not tier-configurable because they are cost-sensitive and quality-insensitive at scale. Tier changes flow through the same rollout process as any other model change (§12), so an enterprise tenant moving to a stronger model still goes through shadow and canary rather than flipping atomically in production.

Everything else is fixed. Tenants cannot change chunking strategy, embedding model, retrieval top-k, reranker, context budget, loop caps, or guardrail thresholds. Those parameters are part of the product's quality contract; letting tenants tune them would fracture the eval surface (§11) into per-tenant configurations we cannot realistically regression-test, and would shift support burden onto our team every time a tenant's custom setting produced a bad answer.

### 6.3 Noisy-neighbor controls

One tenant must not be able to degrade service for others, whether through a buggy integration hammering the API, a promotional spike, or a deliberate abuse attempt. The isolation story in §6.1 is about data; this section is about capacity. We enforce four controls, each targeting a different failure mode.

**Per-tenant request rate limits.** Every workspace has a requests-per-minute cap enforced at the API gateway, keyed by `workspace_id` from the authenticated request. The cap is plan-based: Free tenants get a low ceiling sized for interactive use, Team and Business tenants get headroom for embedded-widget traffic, Enterprise tenants get negotiated limits. Limits are sliding-window, not fixed-bucket, so a tenant cannot burst `2 × limit` across a minute boundary. Rejections return `429` with a `Retry-After` header and are logged to the trace with a distinct `throttled` status so dashboards separate throttled traffic from genuine errors.

**Per-tenant token budgets.** Rate limits cap request count but not cost, and one expensive agent loop can burn more tokens than a hundred FAQ lookups. So we also enforce a rolling 24-hour token budget per tenant, summed across input and output tokens for all models. The budget is advisory on the Free plan (warning only, no block) to avoid user-hostile hard cutoffs during onboarding, and enforced on paid plans with a soft cap that routes further requests to the cheapest model and a hard cap that returns a graceful "daily limit reached, contact your admin" response. Budget usage is queryable via the admin API so customers can wire their own alerting.

**Per-tenant spend caps.** Token budgets are a proxy for cost but not cost itself: model routing, tool calls, and embeddings all affect dollar spend at different rates. So we also track `cost_usd` per tenant per rolling 24 hours from the trace layer (§4.6) and hard-cap it at a configurable per-tenant ceiling. Defaults are set at roughly 3× the 95th percentile of legitimate tenant spend on their plan, so normal workloads never notice the cap and abuse patterns hit it before they become material. Spend cap hits page on-call with the tenant ID so the team can disambiguate "runaway bug in a customer's integration" from "legitimate growth that should raise the cap".

**Fair-share queueing at the LLM client layer.** Rate and budget limits operate per tenant in isolation, but the shared bottleneck is our aggregate LLM provider quota — we have a ceiling of tokens-per-minute with each provider, and if one Enterprise tenant with a generous rate limit saturates it, Free-tier users see elevated latency. The LLM client pool uses a weighted fair-share scheduler: each tenant gets a share of the provider quota proportional to their plan weight, with unused capacity redistributed to whoever has demand. This prevents the rich-get-richer failure mode where a big customer's spike starves everyone else while still letting tenants burst into idle headroom.

We deliberately do not run separate compute clusters per tenant in v1. Dedicated infrastructure is the strongest noisy-neighbor control but the worst for unit economics at our scale, and the four controls above — request rate, token budget, spend cap, fair-share scheduling — close the gap well enough for the tenant sizes we target (§2). Dedicated-pool provisioning is a lever we pull for specific Enterprise contracts where the customer pays for the isolation, not a default tier.

### 6.4 Cost attribution

Every request must be attributable to exactly one tenant for billing, capacity planning, and abuse investigation. The trace layer from §4.6 is the source of truth: no event-emission shortcut, no out-of-band metering, no sampling. Cost attribution uses the same trace stream that powers observability, which guarantees that what we charge for and what we can debug are always the same set of events.

**Trace schema for billing.** Every trace carries `workspace_id`, `request_id`, `route_class` (§8), `model`, `tokens_in`, `tokens_out`, `tool_calls` (list of tool name plus token cost per call), `embedding_tokens`, `reranker_tokens`, `guardrail_tokens`, `cost_usd`, and `timestamp`. `cost_usd` is computed at trace-emit time from a versioned price table keyed by `(model, modality, date)` so retroactive price corrections are possible without re-reading the original payloads. Traces are append-only; a billing correction is a new compensating trace, not an edit.

**Aggregation pipeline.** Traces stream into the data warehouse on a minute-level lag. An hourly job rolls them up into `(workspace_id, hour, route_class, model)` buckets with summed token counts and cost. A daily job rolls hourly into daily, and a monthly job rolls daily into billing periods. Each job is idempotent and keyed by trace `request_id` so reprocessing a backfilled hour does not double-count. The billing period close is a single transaction that freezes a tenant's monthly total and writes it to the invoice table; downstream billing (Stripe, NetSuite) reads from the invoice table, never from the raw traces.

**Billable vs. non-billable.** Not all tokens are tenant-billable. Guardrail calls on rejected requests (prompt-injection block, policy violation) are logged for audit but not charged; charging a tenant for our safety layer is bad policy and incentivizes them to bypass it. Retries caused by our transient infrastructure failures are logged with `retry_reason = infrastructure` and excluded from billing. Retries caused by tenant-side failures (malformed tool arg, user typing correction) are billable. The trace schema carries a boolean `billable` field set at emit time so the aggregation pipeline can filter cleanly without business logic leaking into SQL.

**Audit and dispute surface.** Tenants can query their own usage via an admin API that reads the same aggregation tables billing reads — what they see is exactly what they are charged for, at a one-day lag. Individual traces are queryable by `request_id` for dispute investigation, with content redacted but token counts and costs visible. This removes the "my bill doesn't match my logs" class of support ticket by making the log the bill.

Cost attribution also feeds the spend cap in §6.3: the same aggregation that produces invoices produces real-time per-tenant running totals that the rate limiter consults. Having one accounting source for both billing and throttling means a tenant can never hit a spend cap that is not eventually reflected on their invoice, and can never be billed for usage that did not also count toward their cap.

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
