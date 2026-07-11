"""Data models for request/response logging API."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from luthien_proxy.utils.db import ConnectionProtocol, DatabaseWriteError


@dataclass
class _PendingLog:
    """Accumulates data for a single log row before it is written to DB."""

    direction: str
    transaction_id: str
    session_id: str | None = None
    user_id: str | None = None
    http_method: str | None = None
    url: str | None = None
    request_headers: dict[str, str] | None = None
    request_body: dict[str, Any] | None = None
    response_status: int | None = None
    response_headers: dict[str, str] | None = None
    response_body: dict[str, Any] | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    duration_ms: float | None = None
    model: str | None = None
    is_streaming: bool = False
    endpoint: str | None = None
    error: str | None = None


class _SerializedBodyProtocol(Protocol):
    @property
    def payload(self) -> str | None: ...

    @property
    def size_bytes(self) -> int: ...

    @property
    def truncated(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class _SerializedLogBodySizes:
    request_body_bytes: int
    response_body_bytes: int
    body_truncated: bool


async def insert_log_row(
    conn: ConnectionProtocol,
    pending: _PendingLog,
    serialize_body: Callable[[dict[str, Any] | None], _SerializedBodyProtocol],
) -> _SerializedLogBodySizes:
    """Insert one request_logs row, wrapping driver errors for callers."""
    # SQL avoids the CASE WHEN $N ... $N duplicate-positional pattern (breaks SQLite ?);
    # to_timestamp(NULL) -> NULL on both Postgres and SQLite.
    request_body = serialize_body(pending.request_body)
    response_body = serialize_body(pending.response_body)
    try:
        await conn.execute(
            """
            INSERT INTO request_logs (
                transaction_id, session_id, user_id, direction,
                http_method, url, request_headers, request_body,
                response_status, response_headers, response_body,
                started_at, completed_at, duration_ms,
                model, is_streaming, endpoint, error
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7::jsonb, $8::jsonb,
                $9, $10::jsonb, $11::jsonb,
                to_timestamp($12), to_timestamp($13), $14,
                $15, $16, $17, $18
            )
            """,
            pending.transaction_id,
            pending.session_id,
            pending.user_id,
            pending.direction,
            pending.http_method,
            pending.url,
            json.dumps(pending.request_headers) if pending.request_headers else None,
            request_body.payload,
            pending.response_status,
            json.dumps(pending.response_headers) if pending.response_headers else None,
            response_body.payload,
            pending.started_at,
            pending.completed_at,
            pending.duration_ms,
            pending.model,
            pending.is_streaming,
            pending.endpoint,
            pending.error,
        )
    except Exception as exc:
        raise DatabaseWriteError(
            f"Failed to insert request_log row (direction={pending.direction!r}, "
            f"transaction_id={pending.transaction_id!r}): {exc}",
            cause=exc,
        ) from exc
    return _SerializedLogBodySizes(
        request_body_bytes=request_body.size_bytes,
        response_body_bytes=response_body.size_bytes,
        body_truncated=request_body.truncated or response_body.truncated,
    )


class RequestLogEntry(BaseModel):
    """A single request/response log entry."""

    id: str
    transaction_id: str
    session_id: str | None = None
    user_id: str | None = None
    direction: str
    http_method: str | None = None
    url: str | None = None
    request_headers: dict[str, str] | None = None
    request_body: dict[str, Any] | None = None
    response_status: int | None = None
    response_headers: dict[str, str] | None = None
    response_body: dict[str, Any] | None = None
    started_at: str
    completed_at: str | None = None
    duration_ms: float | None = None
    model: str | None = None
    is_streaming: bool = False
    endpoint: str | None = None
    error: str | None = None


class RequestLogListResponse(BaseModel):
    """Paginated list of request log entries."""

    logs: list[RequestLogEntry]
    total: int
    limit: int
    offset: int


class RequestLogDetailResponse(BaseModel):
    """All log entries (inbound + outbound) for a single transaction."""

    transaction_id: str
    session_id: str | None = None
    user_id: str | None = None
    inbound: RequestLogEntry | None = None
    outbound: RequestLogEntry | None = None


__all__ = [
    "RequestLogEntry",
    "RequestLogListResponse",
    "RequestLogDetailResponse",
    "insert_log_row",
]
