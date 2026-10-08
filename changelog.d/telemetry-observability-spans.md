---
category: Features
---

**OpenTelemetry observability spans and attributes** (#805): adds low-cardinality spans and attributes across credential validation, DB writes, request/response body sizes, and time-to-first-streamed-event, plus an opt-in Sentry init. All attributes are sizes, counts, durations, status, or booleans — no request/response content or credentials are recorded.
  - asyncpg and psycopg queries are traced as spans, and a failed request-log write records its exception on the `request_log.write` span.
  - New `OBSERVABILITY_STDOUT_ENABLED` setting (default `true`, unchanged behaviour) turns off the emitter's full-payload stdout dump; events still reach the database and the event publisher.
