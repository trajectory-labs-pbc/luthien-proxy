"""OpenAI Chat Completions passthrough normalizers."""

from __future__ import annotations

from collections.abc import Mapping

from luthien_proxy.passthrough_materialize.endpoints import EligibleEndpoint, EndpointKind
from luthien_proxy.passthrough_materialize.openai_chat_stream import stream_chat_response
from luthien_proxy.passthrough_materialize.openai_common import (
    PassthroughNormalizeReason,
    canonical_usage,
    error_response,
    fail,
    is_json_object,
    is_json_sequence,
    json_mutable_object,
    json_object_from_string,
    object_field,
    optional_string,
    require_openai_endpoint,
    sequence_field,
    stop_reason,
    text_content_from_openai,
)
from luthien_proxy.passthrough_materialize.payloads import (
    CanonicalRequestInput,
    CanonicalResponseInput,
    JsonMutableObject,
    JsonMutableValue,
    JsonObject,
    JsonValue,
)


def normalize_openai_chat_request(
    endpoint: EligibleEndpoint, request: JsonObject, *, transaction_id: str | None = None
) -> CanonicalRequestInput:
    """Normalize a Chat Completions request into canonical request input."""
    require_openai_endpoint(endpoint, EndpointKind.OPENAI_CHAT_COMPLETIONS, transaction_id)
    model = optional_string(request, "model")
    messages = sequence_field(endpoint, request, "messages", transaction_id)
    final_request: JsonMutableObject = {"model": model, "messages": _chat_messages(endpoint, messages, transaction_id)}
    _copy_optional_request_fields(request, final_request, endpoint, transaction_id)
    stream = request.get("stream") is True
    final_request["stream"] = stream
    return CanonicalRequestInput(
        endpoint=endpoint,
        is_streaming=stream,
        final_model=model,
        original_request=final_request,
        final_request=final_request,
        provider_request=request,
    )


def normalize_openai_chat_response(
    endpoint: EligibleEndpoint,
    response: JsonObject,
    *,
    request_is_streaming: bool,
    http_status: int,
    transaction_id: str | None = None,
) -> CanonicalResponseInput:
    """Normalize a Chat Completions response into canonical response input."""
    require_openai_endpoint(endpoint, EndpointKind.OPENAI_CHAT_COMPLETIONS, transaction_id)
    if http_status >= 400:
        final_response = error_response(http_status, response)
    elif request_is_streaming:
        final_response = stream_chat_response(endpoint, response, transaction_id)
    else:
        final_response = _buffered_chat_response(endpoint, response, transaction_id)
    model_value = final_response.get("model")
    final_model = model_value if isinstance(model_value, str) else None
    return CanonicalResponseInput(
        endpoint=endpoint,
        is_streaming=request_is_streaming,
        final_model=final_model,
        original_response=final_response,
        final_response=final_response,
        provider_response=response,
    )


def _chat_messages(
    endpoint: EligibleEndpoint, messages: JsonValue, transaction_id: str | None
) -> list[JsonMutableValue]:
    result: list[JsonMutableValue] = []
    if not is_json_sequence(messages):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "messages")
    for item in messages:
        if not is_json_object(item):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "message")
        role = item.get("role")
        if not isinstance(role, str):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "message.role")
        result.append(_chat_message(endpoint, role, item, transaction_id))
    return result


def _chat_message(
    endpoint: EligibleEndpoint, role: str, item: Mapping[str, JsonValue], transaction_id: str | None
) -> JsonMutableObject:
    canonical_role = "system" if role == "developer" else role
    match canonical_role:
        case "system" | "user" | "tool":
            content = text_content_from_openai(
                endpoint, item.get("content"), transaction_id, input_prefix="message.content"
            )
            message: JsonMutableObject = {"role": canonical_role, "content": content}
            tool_call_id = item.get("tool_call_id")
            if isinstance(tool_call_id, str):
                message["tool_call_id"] = tool_call_id
            return message
        case "assistant":
            content_blocks: list[JsonMutableValue] = []
            content = item.get("content")
            if content is not None:
                normalized = text_content_from_openai(
                    endpoint, content, transaction_id, input_prefix="assistant.content"
                )
                if isinstance(normalized, str):
                    content_blocks.append({"type": "text", "text": normalized})
                elif isinstance(normalized, list):
                    content_blocks.extend(normalized)
            content_blocks.extend(_tool_calls(endpoint, item.get("tool_calls"), transaction_id))
            return {"role": "assistant", "content": content_blocks}
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "message.role")


def _tool_calls(endpoint: EligibleEndpoint, raw_calls: JsonValue, transaction_id: str | None) -> list[JsonMutableValue]:
    if raw_calls is None:
        return []
    if not is_json_sequence(raw_calls):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "tool_calls")
    calls: list[JsonMutableValue] = []
    for raw_call in raw_calls:
        if not is_json_object(raw_call):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "tool_call")
        function = object_field(endpoint, raw_call, "function", transaction_id)
        name = optional_string(function, "name")
        arguments = optional_string(function, "arguments") or "{}"
        call_id = optional_string(raw_call, "id")
        if not call_id:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "tool_call.id")
        if name is None:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "tool_call.function.name")
        calls.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": json_object_from_string(endpoint, arguments, transaction_id),
            }
        )
    return calls


def _copy_optional_request_fields(
    request: JsonObject, final_request: JsonMutableObject, endpoint: EligibleEndpoint, transaction_id: str | None
) -> None:
    for key in ("tool_choice", "temperature", "top_p", "stop", "max_completion_tokens", "max_tokens"):
        if key in request:
            final_request[key] = json_mutable_object({"value": request[key]})["value"]
    if "max_tokens" not in final_request and isinstance(request.get("max_completion_tokens"), int):
        final_request["max_tokens"] = json_mutable_object({"value": request["max_completion_tokens"]})["value"]
    tools = request.get("tools")
    if tools is not None:
        final_request["tools"] = _tools(endpoint, tools, transaction_id)


def _tools(endpoint: EligibleEndpoint, tools: JsonValue, transaction_id: str | None) -> list[JsonMutableValue]:
    if not is_json_sequence(tools):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "tools")
    result: list[JsonMutableValue] = []
    for tool in tools:
        if not is_json_object(tool) or tool.get("type") != "function":
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "tool")
        function = object_field(endpoint, tool, "function", transaction_id)
        name = optional_string(function, "name")
        if name is None:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "tool.function.name")
        canonical_tool: JsonMutableObject = {
            "name": name,
            "input_schema": json_mutable_object(object_field(endpoint, function, "parameters", transaction_id)),
        }
        description = optional_string(function, "description")
        if description is not None:
            canonical_tool["description"] = description
        result.append(canonical_tool)
    return result


def _buffered_chat_response(
    endpoint: EligibleEndpoint, response: JsonObject, transaction_id: str | None
) -> JsonMutableObject:
    choices = sequence_field(endpoint, response, "choices", transaction_id)
    first = choices[0] if choices else None
    if not is_json_object(first):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "choices[0]")
    message = object_field(endpoint, first, "message", transaction_id)
    return _assistant_response(
        response,
        message,
        optional_string(first, "finish_reason"),
        canonical_usage(response.get("usage")),
        endpoint,
        transaction_id,
    )


def _assistant_response(
    response: JsonObject,
    message: JsonObject,
    finish_reason: str | None,
    usage: JsonMutableObject | None,
    endpoint: EligibleEndpoint,
    transaction_id: str | None,
) -> JsonMutableObject:
    content: list[JsonMutableValue] = []
    text = optional_string(message, "content")
    if text:
        content.append({"type": "text", "text": text})
    refusal = optional_string(message, "refusal")
    if refusal:
        content.append({"type": "text", "text": refusal})
    content.extend(_tool_calls(endpoint, message.get("tool_calls"), transaction_id))
    final: JsonMutableObject = {"role": "assistant", "content": content, "stop_reason": stop_reason(finish_reason)}
    for key in ("id", "model"):
        value = optional_string(response, key)
        if value is not None:
            final[key] = value
    if usage is not None:
        final["usage"] = usage
    return final
