"""OpenAI Responses passthrough normalizers."""

from __future__ import annotations

from collections.abc import Mapping

from luthien_proxy.passthrough_materialize.endpoints import EligibleEndpoint, EndpointKind
from luthien_proxy.passthrough_materialize.openai_common import (
    PassthroughNormalizeReason,
    canonical_usage,
    error_response,
    fail,
    is_json_object,
    is_json_sequence,
    json_mutable,
    json_mutable_object,
    json_object_from_string,
    object_field,
    optional_string,
    require_openai_endpoint,
    text_content_from_openai,
)
from luthien_proxy.passthrough_materialize.openai_responses_stream import fold_response_stream
from luthien_proxy.passthrough_materialize.payloads import (
    CanonicalRequestInput,
    CanonicalResponseInput,
    JsonMutableObject,
    JsonMutableValue,
    JsonObject,
    JsonValue,
)


def normalize_openai_responses_request(
    endpoint: EligibleEndpoint, request: JsonObject, *, transaction_id: str | None = None
) -> CanonicalRequestInput:
    """Normalize an OpenAI Responses request into canonical request input."""
    require_openai_endpoint(endpoint, EndpointKind.OPENAI_RESPONSES, transaction_id)
    model = optional_string(request, "model")
    messages: list[JsonMutableValue] = []
    instructions = optional_string(request, "instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": instructions})
    messages.extend(_input_messages(endpoint, request.get("input"), transaction_id))
    final_request: JsonMutableObject = {"model": model, "messages": messages}
    _copy_request_fields(request, final_request, endpoint, transaction_id)
    stream = request.get("stream") is True
    final_request["stream"] = stream
    return CanonicalRequestInput(endpoint, stream, model, final_request, final_request, request)


def normalize_openai_responses_response(
    endpoint: EligibleEndpoint,
    response: JsonObject,
    *,
    request_is_streaming: bool,
    http_status: int,
    transaction_id: str | None = None,
) -> CanonicalResponseInput:
    """Normalize an OpenAI Responses response into canonical response input."""
    require_openai_endpoint(endpoint, EndpointKind.OPENAI_RESPONSES, transaction_id)
    if http_status >= 400:
        final_response = error_response(http_status, response)
    elif request_is_streaming:
        final_response = _stream_response(endpoint, response, transaction_id)
    else:
        final_response = _buffered_response(endpoint, response, transaction_id)
    model_value = final_response.get("model")
    final_model = model_value if isinstance(model_value, str) else None
    return CanonicalResponseInput(endpoint, request_is_streaming, final_model, final_response, final_response, response)


def _input_messages(
    endpoint: EligibleEndpoint, raw_input: JsonValue, transaction_id: str | None
) -> list[JsonMutableValue]:
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if not is_json_sequence(raw_input):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "input")
    result: list[JsonMutableValue] = []
    for item in raw_input:
        if not is_json_object(item):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "input item")
        result.append(_input_item(endpoint, item, transaction_id))
    return result


def _input_item(
    endpoint: EligibleEndpoint, item: Mapping[str, JsonValue], transaction_id: str | None
) -> JsonMutableValue:
    item_type = item.get("type")
    match item_type:
        case "function_call_output":
            call_id = optional_string(item, "call_id")
            output = optional_string(item, "output")
            if call_id is None or output is None:
                fail(
                    endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "function_call_output"
                )
            return {"role": "tool", "tool_call_id": call_id, "content": output}
        case None:
            role = optional_string(item, "role")
            if role is None:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "input.role")
            content = text_content_from_openai(
                endpoint, item.get("content"), transaction_id, input_prefix="input.content"
            )
            return {"role": "system" if role == "developer" else role, "content": content}
        case str():
            fail(
                endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, f"input item.type:{item_type}"
            )
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "input item.type")


def _copy_request_fields(
    request: JsonObject, final_request: JsonMutableObject, endpoint: EligibleEndpoint, transaction_id: str | None
) -> None:
    for key in ("tool_choice", "temperature", "top_p", "max_output_tokens"):
        if key in request:
            final_request[key] = json_mutable(request[key])
    if isinstance(request.get("max_output_tokens"), int):
        final_request["max_tokens"] = json_mutable(request["max_output_tokens"])
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
        name = optional_string(tool, "name")
        if name is None:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "tool.name")
        canonical: JsonMutableObject = {
            "name": name,
            "input_schema": json_mutable_object(object_field(endpoint, tool, "parameters", transaction_id)),
        }
        description = optional_string(tool, "description")
        if description is not None:
            canonical["description"] = description
        result.append(canonical)
    return result


def _buffered_response(
    endpoint: EligibleEndpoint, response: JsonObject, transaction_id: str | None
) -> JsonMutableObject:
    content: list[JsonMutableValue] = []
    output = response.get("output")
    if not is_json_sequence(output):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "output")
    for item in output:
        if not is_json_object(item):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "output item")
        content.extend(_output_item(endpoint, item, transaction_id))
    final = _response_base(response, content)
    _copy_status_fields(response, final)
    return final


def _output_item(
    endpoint: EligibleEndpoint, item: Mapping[str, JsonValue], transaction_id: str | None
) -> list[JsonMutableValue]:
    match item.get("type"):
        case "message":
            content = item.get("content")
            if not is_json_sequence(content):
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "message.content")
            blocks: list[JsonMutableValue] = []
            for block in content:
                if not is_json_object(block):
                    fail(
                        endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "message.content block"
                    )
                blocks.extend(_output_content(endpoint, block, transaction_id))
            return blocks
        case "function_call":
            call_id = optional_string(item, "call_id")
            name = optional_string(item, "name")
            arguments = optional_string(item, "arguments") or "{}"
            if not call_id:
                fail(
                    endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "function_call.call_id"
                )
            if name is None:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "function_call.name")
            return [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": json_object_from_string(endpoint, arguments, transaction_id),
                }
            ]
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "output item")


def _output_content(
    endpoint: EligibleEndpoint, block: Mapping[str, JsonValue], transaction_id: str | None
) -> list[JsonMutableValue]:
    match block.get("type"):
        case "output_text":
            text = optional_string(block, "text")
            if text is None:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "output_text")
            return [{"type": "text", "text": text}]
        case "refusal":
            refusal = optional_string(block, "refusal")
            if refusal is None:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "refusal")
            return [{"type": "text", "text": refusal}]
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "output content")


def _stream_response(endpoint: EligibleEndpoint, response: JsonObject, transaction_id: str | None) -> JsonMutableObject:
    folded = fold_response_stream(endpoint, response, transaction_id)
    completed = folded.completed or {}
    if is_json_sequence(completed.get("output")):
        return _buffered_response(endpoint, completed, transaction_id)
    final = _response_base(completed, [{"type": "text", "text": folded.text}])
    _copy_status_fields(completed, final)
    return final


def _response_base(response: JsonObject, content: list[JsonMutableValue]) -> JsonMutableObject:
    stop = (
        "tool_use" if any(is_json_object(item) and item.get("type") == "tool_use" for item in content) else "end_turn"
    )
    final: JsonMutableObject = {"role": "assistant", "content": content, "stop_reason": stop}
    for key in ("id", "model"):
        value = optional_string(response, key)
        if value is not None:
            final[key] = value
    usage = canonical_usage(response.get("usage"))
    if usage is not None:
        final["usage"] = usage
    return final


def _copy_status_fields(response: JsonObject, final: JsonMutableObject) -> None:
    for key in ("status", "incomplete_details", "error"):
        if key in response:
            final[key] = json_mutable(response[key])
    if response.get("status") in ("failed", "incomplete") or response.get("error") is not None:
        final["stop_reason"] = "error"
