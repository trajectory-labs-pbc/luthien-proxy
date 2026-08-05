---
category: Fixes
pr: 807
---

Stop Sentry capturing `opentelemetry.context` "Failed to detach context" log
noise. OTel swallows the error itself and the proxied request is unaffected,
but via the logging integration it fired on ~every streaming request and
exhausted the Sentry error quota within hours of enabling Sentry in
production.
