---
category: Features
---
**Configurable Anthropic upstream (`ANTHROPIC_BASE_URL`)**: the proxied `/v1/messages` path, the raw `/v1/*` passthrough, credential validation, and judge calls made with the request's own credential all go to one configured base URL (default `https://api.anthropic.com`), rejected at startup if it is not an absolute http(s) URL
  - Lets the proxy sit in front of an LLM gateway that issues its own bearer tokens (e.g. hawk middleman): passthrough clients were built on the SDK default and validation always hit `api.anthropic.com`, so a gateway-issued token was forwarded to the wrong host and rejected
  - The variable is the one the Anthropic SDK already reads, so mock and gateway upstreams are configured once; `LLM_JUDGE_API_BASE` still overrides it for judge traffic
  - `scripts/start_mock_gateway.py` re-reads settings after exporting the mock URL, matching the sqlite e2e boot
