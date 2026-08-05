---
category: Fixes
pr: 806
---

**SQLite timestamps**: Store `datetime` bind parameters as ISO-8601 with a `T` separator so `created_at` range filters compare correctly on SQLite. Previously a `datetime` bind was stored with a space separator (stdlib `sqlite3`'s legacy adapter), which sorts before the `T` form used by `.isoformat()` comparisons, silently breaking history pagination deltas, session-search time ranges, and retention cutoffs.
