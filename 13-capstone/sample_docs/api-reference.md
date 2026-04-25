# NovaCRM — API Reference

## Overview

The NovaCRM REST API lets you read and write CRM data programmatically. Available on Pro and Enterprise plans. Base URL: `https://api.novacrm.com/v1`.

All requests must include an API key in the `Authorization` header:

```
Authorization: Bearer ncrm_sk_live_abc123...
```

## Authentication

Generate an API key at **Settings > API Keys** in the NovaCRM dashboard. You can create multiple keys with different permissions (read-only, read-write, admin). Each key is scoped to one workspace.

- API keys start with `ncrm_sk_live_` (production) or `ncrm_sk_test_` (sandbox)
- Test keys operate on a separate sandbox dataset — safe for development
- Rotate keys at any time; the old key stops working immediately
- Never embed API keys in client-side code or public repositories

## Rate limits

Rate limits are per API key, measured in requests per minute (RPM):

| Plan | RPM | Burst (per second) |
|------|-----|--------------------|
| Pro | 1,000 | 50 |
| Enterprise | 10,000 | 200 |

When you exceed the limit, the API returns `429 Too Many Requests` with a `Retry-After` header (in seconds). Implement exponential backoff in your client.

## Common endpoints

### Contacts

- `GET /v1/contacts` — List contacts (paginated, 100 per page)
- `GET /v1/contacts/:id` — Get a single contact
- `POST /v1/contacts` — Create a contact
- `PATCH /v1/contacts/:id` — Update a contact
- `DELETE /v1/contacts/:id` — Delete a contact (soft delete, recoverable for 30 days)

### Deals

- `GET /v1/deals` — List deals (filterable by pipeline, stage, owner)
- `GET /v1/deals/:id` — Get a single deal
- `POST /v1/deals` — Create a deal
- `PATCH /v1/deals/:id` — Update a deal (including stage changes)
- `DELETE /v1/deals/:id` — Delete a deal

### Pipelines

- `GET /v1/pipelines` — List all pipelines with stages
- `POST /v1/pipelines` — Create a pipeline (Pro: max 10, Enterprise: unlimited)

### Activities

- `GET /v1/activities` — List activities (calls, emails, notes) for a contact or deal
- `POST /v1/activities` — Log an activity

## Webhooks

Enterprise plans can register webhooks to receive real-time events:

```
POST /v1/webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["contact.created", "deal.stage_changed", "deal.won", "deal.lost"]
}
```

Webhook payloads are signed with HMAC-SHA256. Verify the `X-NovaCRM-Signature` header using your webhook secret (found in Settings > Webhooks).

Events are delivered at least once. Your endpoint should be idempotent. If delivery fails, we retry 3 times with exponential backoff (1 min, 5 min, 30 min). After 3 failures the webhook is disabled and you receive an email notification.

## Error codes

| Code | Meaning | Common cause |
|------|---------|--------------|
| 400 | Bad Request | Missing required field, invalid JSON |
| 401 | Unauthorized | Missing or invalid API key |
| 403 | Forbidden | API key lacks permission for this action |
| 404 | Not Found | Resource doesn't exist or was deleted |
| 409 | Conflict | Duplicate resource (e.g., contact with same email) |
| 422 | Unprocessable Entity | Validation error (e.g., email format invalid) |
| 429 | Too Many Requests | Rate limit exceeded — check Retry-After header |
| 500 | Internal Server Error | Our fault — retry after a few seconds, contact support if persistent |

All error responses include a JSON body:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "You have exceeded the rate limit of 1000 requests per minute.",
    "retry_after": 12
  }
}
```

## Pagination

List endpoints use cursor-based pagination:

```
GET /v1/contacts?limit=50&after=cursor_abc123
```

The response includes a `next_cursor` field. Pass it as the `after` parameter to get the next page. When `next_cursor` is null, you've reached the end.

## Sandbox environment

Use your test API key (`ncrm_sk_test_...`) to work against the sandbox. The sandbox has its own dataset and does not affect production data. Sandbox data resets weekly on Sunday at midnight UTC. The sandbox has the same rate limits as your plan.
