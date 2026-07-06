---
category: Features
pr: 796
---

**Multi-provider passthrough capture**: capture OpenAI (`/openai/*`) and Gemini (`/gemini/*`) passthrough traffic into `request_logs`, mirroring the existing Anthropic `/v1/*` capture.
  - Streaming and non-streaming responses are recorded; payloads are sanitized before persistence.
  - Cross-provider session grouping via the existing header/metadata contract, so a single logical conversation is retrievable regardless of provider.
