---
category: Fixes
---

**Expected upstream provider errors no longer burn Sentry quota**: the Sentry Anthropic integration captures provider errors unhandled at the SDK call site, before the pipeline converts them into a `BackendAPIError` response. `_sentry_before_send` now drops throttling/availability errors (408, 429, 500, 502, 503, 504, 529) as well as 400/404/401 — cases where the client sent content, a model name, or a credential that Anthropic legitimately rejected and the proxy transparently relayed. The issue stream now shows proxy defects rather than the backend's own responses to the client's or provider's conditions. A status code outside this set (e.g. 403) still reports, so a new upstream failure mode stays visible.
