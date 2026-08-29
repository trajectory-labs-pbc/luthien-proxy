---
category: Fixes
---

**Client/network-lifecycle noise no longer burns Sentry quota**: `starlette.requests.ClientDisconnect` (the client walked away mid-body-read; always unhandled, never a proxy defect) is now dropped in `_sentry_before_send`. Mid-stream `httpx.TransportError`s (`RemoteProtocolError`, `ReadTimeout`, `ReadError`, ...) raised while reading a streaming upstream response now log at warning instead of error, matching the existing `AnthropicConnectionError` handling. Malformed JSON in an incoming request (a client error, already returned to the client as an HTTP 400) now logs at warning instead of error. Genuine proxy bugs mid-stream still log at error and still report.
