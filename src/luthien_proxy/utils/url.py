"""URL sanitization utilities for safe logging."""

from __future__ import annotations

from urllib.parse import urlparse, urlsplit, urlunparse


def sanitize_url_for_logging(url: str) -> str:
    """Strip credentials from a URL so it can be safely logged.

    Replaces the password (and optionally username) portion of the URL
    with ``***`` while preserving host, port, path, and query parameters.

    Returns the original string unchanged if it cannot be parsed.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    if not parsed.password:
        return url

    # Rebuild netloc: keep username visible but mask the password
    if parsed.username:
        masked_netloc = f"{parsed.username}:***@{parsed.hostname}"
    else:
        masked_netloc = f"***@{parsed.hostname}"

    if parsed.port:
        masked_netloc += f":{parsed.port}"

    return urlunparse((parsed.scheme, masked_netloc, parsed.path, parsed.params, parsed.query, ""))


def validate_anthropic_base_url(value: str) -> None:
    """Fail startup on an ANTHROPIC_BASE_URL that cannot carry requests.

    An empty or scheme-less value is not a misconfiguration the gateway can
    survive: the SDK client would fall back to its default host while the raw
    /v1 passthrough and credential validation build relative URLs, so one
    request's credential could be judged by one host and forwarded to another.
    """
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"ANTHROPIC_BASE_URL must be an absolute http(s) URL with a host, got {value!r}")
