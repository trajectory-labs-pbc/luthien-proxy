---
category: Fixes
---

**Expected upstream provider errors no longer burn Sentry quota**: the Sentry Anthropic integration captures provider throttling and availability errors (429, 529, 5xx) unhandled at the SDK call site, before the pipeline converts them into a `BackendAPIError` response. `_sentry_before_send` now drops those, so the issue stream shows proxy defects rather than the backend's normal backpressure. Client errors such as 400 still report, since a malformed request is actionable.
