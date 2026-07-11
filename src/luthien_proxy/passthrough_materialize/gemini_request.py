"""Gemini generateContent request normalization."""

from __future__ import annotations

from collections.abc import Mapping

from luthien_proxy.passthrough_materialize.endpoints import EligibleEndpoint
from luthien_proxy.passthrough_materialize.gemini_common import (
    gemini_endpoint_streaming,
    gemini_function_call,
    gemini_function_call_id,
    gemini_model_from_path,
)
from luthien_proxy.passthrough_materialize.openai_common import (
    PassthroughNormalizeReason,
    fail,
    is_json_object,
    is_json_sequence,
    json_mutable_object,
    optional_string,
    sequence_field,
)
from luthien_proxy.passthrough_materialize.payloads import (
    CanonicalRequestInput,
    JsonMutableObject,
    JsonMutableValue,
    JsonObject,
    JsonValue,
)


def normalize_gemini_request(
    endpoint: EligibleEndpoint, request: JsonObject, *, transaction_id: str | None = None
) -> CanonicalRequestInput:
    """Normalize a Gemini generateContent request into canonical request input."""
    stream = gemini_endpoint_streaming(endpoint, transaction_id)
    messages = _request_messages(endpoint, request, transaction_id)
    model = gemini_model_from_path(endpoint.path)
    final_request: JsonMutableObject = {"model": model, "messages": messages, "stream": stream}
    _copy_request_configuration(endpoint, request, final_request, transaction_id)
    return CanonicalRequestInput(endpoint, stream, model, final_request, final_request, request)


def _request_messages(
    endpoint: EligibleEndpoint, request: JsonObject, transaction_id: str | None
) -> list[JsonMutableValue]:
    messages: list[JsonMutableValue] = []
    system_instruction = request.get("systemInstruction")
    if system_instruction is not None:
        if not is_json_object(system_instruction):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "systemInstruction")
        messages.append({"role": "system", "content": _parts(endpoint, system_instruction, "system", transaction_id)})
    contents = sequence_field(endpoint, request, "contents", transaction_id)
    for content in contents:
        if not is_json_object(content):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "contents item")
        messages.append(_content_message(endpoint, content, transaction_id))
    return messages


def _content_message(endpoint: EligibleEndpoint, content: JsonObject, transaction_id: str | None) -> JsonMutableObject:
    role = optional_string(content, "role")
    match role:
        case "user":
            return {"role": "user", "content": _parts(endpoint, content, "user", transaction_id)}
        case "model":
            return {"role": "assistant", "content": _parts(endpoint, content, "model", transaction_id)}
        case None:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "contents.role")
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "contents.role")


def _parts(
    endpoint: EligibleEndpoint, content: JsonObject, role: str, transaction_id: str | None
) -> list[JsonMutableValue]:
    parts = sequence_field(endpoint, content, "parts", transaction_id)
    blocks: list[JsonMutableValue] = []
    function_ordinal = 0
    for part in parts:
        if not is_json_object(part):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, f"{role}.part")
        blocks.append(_part(endpoint, part, role, function_ordinal, transaction_id))
        if "functionCall" in part or "functionResponse" in part:
            function_ordinal += 1
    return blocks


def _part(
    endpoint: EligibleEndpoint, part: JsonObject, role: str, function_ordinal: int, transaction_id: str | None
) -> JsonMutableValue:
    match part:
        case {"text": str() as text}:
            return {"type": "text", "text": text}
        case {"functionCall": Mapping() as call} if role == "model":
            return gemini_function_call(endpoint, call, function_ordinal, transaction_id)
        case {"functionResponse": Mapping() as response} if role == "user":
            return _function_response(endpoint, response, function_ordinal, transaction_id)
        case {"functionCall": _} | {"functionResponse": _}:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, f"{role}.part")
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, f"{role}.part")


def _function_response(
    endpoint: EligibleEndpoint, response: Mapping[str, JsonValue], ordinal: int, transaction_id: str | None
) -> JsonMutableObject:
    name = optional_string(response, "name")
    response_value = response.get("response")
    if name is None:
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "functionResponse.name")
    if not is_json_object(response_value):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "functionResponse.response")
    return {
        "type": "tool_result",
        "tool_use_id": gemini_function_call_id(response, name, ordinal),
        "content": json_mutable_object(response_value),
    }


def _copy_request_configuration(
    endpoint: EligibleEndpoint,
    request: JsonObject,
    final_request: JsonMutableObject,
    transaction_id: str | None,
) -> None:
    tools = request.get("tools")
    if tools is not None:
        final_request["tools"] = _tools(endpoint, tools, transaction_id)
    tool_config = request.get("toolConfig")
    if tool_config is not None:
        final_request["tool_choice"] = _tool_choice(endpoint, tool_config, transaction_id)
    generation_config = request.get("generationConfig")
    if generation_config is not None:
        if not is_json_object(generation_config):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "generationConfig")
        final_request["generation_config"] = json_mutable_object(generation_config)
        _generation_config(endpoint, generation_config, final_request, transaction_id)


def _tools(endpoint: EligibleEndpoint, raw_tools: JsonValue, transaction_id: str | None) -> list[JsonMutableValue]:
    if not is_json_sequence(raw_tools):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "tools")
    tools: list[JsonMutableValue] = []
    for tool in raw_tools:
        match tool:
            case {"functionDeclarations": _} if is_json_object(tool):
                declarations = sequence_field(endpoint, tool, "functionDeclarations", transaction_id)
                for declaration in declarations:
                    if not is_json_object(declaration):
                        fail(
                            endpoint,
                            transaction_id,
                            PassthroughNormalizeReason.MALFORMED_PAYLOAD,
                            "functionDeclaration",
                        )
                    tools.append(_function_declaration(endpoint, declaration, transaction_id))
            case _:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "tool")
    return tools


def _function_declaration(
    endpoint: EligibleEndpoint, declaration: JsonObject, transaction_id: str | None
) -> JsonMutableObject:
    name = optional_string(declaration, "name")
    if name is None:
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MISSING_REQUIRED_FIELD, "functionDeclaration.name")
    result: JsonMutableObject = {"name": name}
    description = optional_string(declaration, "description")
    if description is not None:
        result["description"] = description
    parameters = declaration.get("parameters")
    if parameters is not None:
        if not is_json_object(parameters):
            fail(
                endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "functionDeclaration.parameters"
            )
        result["input_schema"] = json_mutable_object(parameters)
    return result


def _tool_choice(endpoint: EligibleEndpoint, raw_config: JsonValue, transaction_id: str | None) -> JsonMutableObject:
    if not is_json_object(raw_config):
        fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "toolConfig")
    function_config = raw_config.get("functionCallingConfig")
    if not is_json_object(function_config):
        fail(
            endpoint,
            transaction_id,
            PassthroughNormalizeReason.MISSING_REQUIRED_FIELD,
            "toolConfig.functionCallingConfig",
        )
    mode = optional_string(function_config, "mode")
    match mode:
        case "AUTO":
            result: JsonMutableObject = {"mode": "auto"}
        case "ANY":
            result = {"mode": "any"}
        case "NONE":
            result = {"mode": "none"}
        case "VALIDATED":
            result = {"mode": "validated"}
        case None:
            fail(
                endpoint,
                transaction_id,
                PassthroughNormalizeReason.MISSING_REQUIRED_FIELD,
                "functionCallingConfig.mode",
            )
        case _:
            fail(endpoint, transaction_id, PassthroughNormalizeReason.UNSUPPORTED_VARIANT, "functionCallingConfig.mode")
    names = function_config.get("allowedFunctionNames")
    if names is not None:
        if not is_json_sequence(names):
            fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "allowedFunctionNames")
        allowed_names: list[JsonMutableValue] = []
        for name in names:
            if not isinstance(name, str):
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, "allowedFunctionNames")
            allowed_names.append(name)
        result["allowed_function_names"] = allowed_names
    return result


def _generation_config(
    endpoint: EligibleEndpoint,
    config: JsonObject,
    final_request: JsonMutableObject,
    transaction_id: str | None,
) -> None:
    for key, value in config.items():
        match key, value:
            case "temperature", int() | float() if not isinstance(value, bool):
                final_request["temperature"] = value
            case "topP", int() | float() if not isinstance(value, bool):
                final_request["top_p"] = value
            case "maxOutputTokens", int() if not isinstance(value, bool):
                final_request["max_tokens"] = value
            case "candidateCount", int() if not isinstance(value, bool):
                final_request["candidate_count"] = value
            case "stopSequences", _ if is_json_sequence(value):
                stop_sequences: list[JsonMutableValue] = []
                for item in value:
                    if not isinstance(item, str):
                        fail(
                            endpoint,
                            transaction_id,
                            PassthroughNormalizeReason.MALFORMED_PAYLOAD,
                            "generationConfig.stopSequences",
                        )
                    stop_sequences.append(item)
                final_request["stop"] = stop_sequences
            case "temperature" | "topP" | "maxOutputTokens" | "candidateCount" | "stopSequences", _:
                fail(endpoint, transaction_id, PassthroughNormalizeReason.MALFORMED_PAYLOAD, f"generationConfig.{key}")
            case _:
                continue


__all__ = ["normalize_gemini_request"]
