"""Shared private helpers for OpenAI passthrough normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, TypeGuard

from luthien_proxy.passthrough_materialize.endpoints import EligibleEndpoint, EndpointKind, Provider
from luthien_proxy.passthrough_materialize.payloads import JsonMutableObject, JsonMutableValue, JsonObject, JsonValue


class PassthroughNormalizeReason(StrEnum):
    """Stable reason codes for retryable passthrough normalization failures."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MALFORMED_JSON = "malformed_json"
    MALFORMED_PAYLOAD = "malformed_payload"
    UNSUPPORTED_ENDPOINT = "unsupported_endpoint"
    UNSUPPORTED_VARIANT = "unsupported_variant"
    CAPTURE_TRUNCATED = "capture_truncated"


@dataclass(frozen=True, slots=True)
class PassthroughNormalizeError(Exception):
    """Typed failure raised when eligible passthrough JSON cannot become canonical."""

    provider: Provider
    endpoint_kind: EndpointKind
    endpoint_path: str
    transaction_id: str | None
    reason: PassthroughNormalizeReason
    detail: str

    def __str__(self) -> str:
        """Return a stable diagnostic string for logs and reports."""
        tx = self.transaction_id or "<unknown>"
        return f"{self.provider.value}:{self.endpoint_kind.value}:{tx}:{self.reason.value}:{self.detail}"


def fail(
    endpoint: EligibleEndpoint, transaction_id: str | None, reason: PassthroughNormalizeReason, detail: str
) -> NoReturn:
    """Raise a typed passthrough normalization error."""
    raise PassthroughNormalizeError(
        provider=endpoint.provider,
        endpoint_kind=endpoint.kind,
        endpoint_path=endpoint.path,
        transaction_id=transaction_id,
        reason=reason,
        detail=detail,
    )


def is_json_object(value: JsonValue) -> TypeGuard[JsonObject]:
    """Return whether a JSON value is an object."""
    return isinstance(value, Mapping)


def is_json_sequence(value: JsonValue) -> TypeGuard[Sequence[JsonValue]]:
    """Return whether a JSON value is a non-string array."""
    return isinstance(value, Sequence) and not isinstance(value, str)


def require_openai_endpoint(endpoint: EligibleEndpoint, kind: EndpointKind, transaction_id: str | None) -> None:
    """Ensure a normalizer is used with its matching OpenAI endpoint kind."""
    if endpoint.provider != Provider.OPENAI or endpoint.kind != kind:
        fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_ENDPOINT, "endpoint kind mismatch")


def object_field(endpoint: EligibleEndpoint, value: JsonObject, key: str, transaction_id: str | None) -> JsonObject:
    """Return a required JSON object field or raise a typed error."""
    item = value.get(key)
    if is_json_object(item):
        return item
    fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, key)


def sequence_field(
    endpoint: EligibleEndpoint, value: JsonObject, key: str, transaction_id: str | None
) -> Sequence[JsonValue]:
    """Return a required JSON array field or raise a typed error."""
    item = value.get(key)
    if is_json_sequence(item):
        return item
    fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, key)


def optional_string(value: JsonObject, key: str) -> str | None:
    """Return an optional JSON string field."""
    item = value.get(key)
    return item if isinstance(item, str) else None


def json_object_from_string(endpoint: EligibleEndpoint, raw: str, transaction_id: str | None) -> JsonMutableObject:
    """Parse a JSON object encoded as a string or raise a typed error."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_JSON, "function arguments")
    if is_json_object(parsed):
        return json_mutable_object(parsed)
    fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_JSON, "function arguments not object")


def json_mutable_object(value: Mapping[str, JsonValue]) -> JsonMutableObject:
    """Deep-copy a JSON object into mutable JSON-compatible containers."""
    return {key: json_mutable(item) for key, item in value.items()}


def json_mutable(value: JsonValue) -> JsonMutableValue:
    """Deep-copy a JSON value into mutable JSON-compatible containers."""
    match value:
        case None | str() | bool() | int() | float():
            return value
        case Mapping():
            return json_mutable_object(value)
        case Sequence():
            return [json_mutable(item) for item in value]


def text_content_from_openai(
    endpoint: EligibleEndpoint, content: JsonValue, transaction_id: str | None, *, input_prefix: str
) -> JsonMutableValue:
    """Normalize OpenAI text-only content into history-compatible content."""
    match content:
        case str():
            return content
        case Sequence() if not isinstance(content, str):
            blocks: list[JsonMutableValue] = []
            for block in content:
                if not is_json_object(block):
                    fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "content block")
                block_type = block.get("type")
                match block_type:
                    case "text" | "input_text" | "output_text":
                        text = block.get("text")
                        if not isinstance(text, str):
                            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "text block")
                        blocks.append({"type": "text", "text": text})
                    case _:
                        fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, input_prefix)
            return blocks
        case None:
            return ""
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, input_prefix)


def canonical_usage(usage: JsonValue) -> JsonMutableObject | None:
    """Map OpenAI token counters into canonical usage counters."""
    if not isinstance(usage, Mapping):
        return None
    input_tokens = _first_int(usage, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(usage, "completion_tokens", "output_tokens")
    result: JsonMutableObject = {}
    if isinstance(input_tokens, int):
        result["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        result["output_tokens"] = output_tokens
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    return result or None


def _first_int(usage: Mapping[str, JsonValue], first_key: str, second_key: str) -> int | None:
    first = usage.get(first_key)
    if isinstance(first, int):
        return first
    second = usage.get(second_key)
    return second if isinstance(second, int) else None


def error_response(http_status: int, response: JsonObject) -> JsonMutableObject:
    """Build a canonical upstream-error response without synthetic text."""
    return {
        "role": "assistant",
        "content": [],
        "stop_reason": "error",
        "error": {"status_code": http_status, "body": json_mutable_object(response)},
    }


def ensure_not_truncated(endpoint: EligibleEndpoint, response: JsonObject, transaction_id: str | None) -> None:
    """Reject known-truncated stream captures before canonicalization."""
    if response.get("capture_truncated") is True or "raw" in response:
        fail(endpoint, transaction_id, PassthroughNormalizeReason.CAPTURE_TRUNCATED, "stream capture truncated")


def stop_reason(reason: str | None) -> str:
    """Map OpenAI finish reasons to canonical stop reasons."""
    match reason:
        case "tool_calls":
            return "tool_use"
        case "length":
            return "max_tokens"
        case "stop" | "content_filter" | None:
            return "end_turn"
        case _:
            return "end_turn"
