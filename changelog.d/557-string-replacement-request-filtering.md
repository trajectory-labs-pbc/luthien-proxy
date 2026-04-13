---
category: Features
---

**StringReplacementPolicy request-side filtering**: Added `apply_to` config option (`"request"`, `"response"`, `"both"`; default `"response"`) enabling prompt injection defense by stripping patterns from incoming request messages — including tool result content — before they reach the model. OTEL span events track intervention counts (`blocks_modified`, `total_replacements`) when content is modified. Existing configs are unaffected (default is `"response"`).
