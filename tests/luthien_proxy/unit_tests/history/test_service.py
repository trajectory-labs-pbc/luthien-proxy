"""Tests for conversation history service layer.

Tests the pure business logic functions for fetching sessions,
parsing conversation turns, and exporting to markdown.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.constants import DEFAULT_TEST_MODEL

from luthien_proxy.history.models import (
    ConversationMessage,
    ConversationTurn,
    MessageType,
    PolicyAnnotation,
    SessionDetail,
    SessionSearchParams,
)
from luthien_proxy.history.service import (
    CallEventRange,
    StoredEvent,
    _build_turn,
    _extract_preview_message,
    _extract_tool_calls,
    _get_event_summary,
    _parse_request_messages,
    _parse_response_messages,
    _safe_parse_json,
    export_session_jsonl,
    export_session_markdown,
    extract_text_content,
    fetch_session_detail,
    fetch_session_list,
    iter_session_turns,
)


async def _collect_bytes(chunks) -> bytes:
    parts: list[bytes] = []
    async for chunk in chunks:
        if isinstance(chunk, str):
            parts.append(chunk.encode())
        else:
            parts.append(chunk)
    return b"".join(parts)


def _enable_mock_transaction(mock_conn: AsyncMock) -> None:
    mock_conn.transaction = MagicMock()
    mock_conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)


def _equivalence_rows() -> list[dict[str, object]]:
    return [
        {
            "call_id": "call-1",
            "event_type": "transaction.request_recorded",
            "payload": {
                "final_model": "gpt-4",
                "original_request": {"messages": [{"role": "user", "content": "Hi"}]},
                "final_request": {"messages": [{"role": "user", "content": "Hi"}]},
            },
            "created_at": datetime(2025, 1, 15, 10, 0, 0),
        },
        {
            "call_id": "call-1",
            "event_type": "transaction.streaming_response_recorded",
            "payload": {
                "original_response": {"choices": [{"message": {"content": "Hello!"}}]},
                "final_response": {"choices": [{"message": {"content": "Hello!"}}]},
            },
            "created_at": datetime(2025, 1, 15, 10, 0, 1),
        },
        {
            "call_id": "call-2",
            "event_type": "transaction.request_recorded",
            "payload": {
                "final_model": "claude-3-sonnet",
                "original_request": {"messages": [{"role": "user", "content": "Hi"}]},
                "final_request": {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                        {"role": "user", "content": "Use the tool"},
                    ]
                },
            },
            "created_at": datetime(2025, 1, 15, 10, 2, 0),
        },
        {
            "call_id": "call-2",
            "event_type": "policy.anthropic_judge.tool_call_blocked",
            "payload": {"summary": "Dangerous operation blocked", "rule": "deny"},
            "created_at": datetime(2025, 1, 15, 10, 2, 1),
        },
        {
            "call_id": "call-2",
            "event_type": "transaction.non_streaming_response_recorded",
            "payload": {
                "original_response": {"choices": [{"message": {"content": "Done"}}]},
                "final_response": {"choices": [{"message": {"content": "Blocked"}}]},
            },
            "created_at": datetime(2025, 1, 15, 10, 2, 2),
        },
    ]


class TestGetEventSummary:
    """Test friendly-text fallback for known policy event types."""

    @pytest.mark.parametrize(
        "event_type,expected",
        [
            (
                "policy.string_replacement.request_modified",
                "Request modified by string replacement",
            ),
            (
                "policy.string_replacement.response_modified",
                "Response modified by string replacement",
            ),
            ("policy.judge.tool_call_blocked", "Tool call blocked"),
        ],
    )
    def test_falls_back_to_event_type_description(self, event_type, expected):
        """When payload has no `summary`, use the dict-based description."""
        assert _get_event_summary(event_type, None) == expected
        assert _get_event_summary(event_type, {}) == expected
        assert _get_event_summary(event_type, {"summary": ""}) == expected

    def test_payload_summary_takes_precedence(self):
        """A non-empty payload `summary` wins over the fallback dict."""
        assert (
            _get_event_summary(
                "policy.string_replacement.response_modified",
                {"summary": "Replaced 'foo' with 'bar'"},
            )
            == "Replaced 'foo' with 'bar'"
        )

    def test_unknown_event_type_returns_raw(self):
        """Unknown event types fall through to the raw event_type string."""
        assert _get_event_summary("policy.unknown.event", None) == "policy.unknown.event"


class TestExtractTextContent:
    """Test text content extraction from various message formats."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("Hello world", "Hello world"),
            ("", ""),
            (None, ""),
            ([{"type": "text", "text": "First"}], "First"),
            ([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}], "A\nB"),
            ([{"type": "image", "url": "http://..."}], ""),
            ([{"type": "text", "text": "Text"}, {"type": "tool_use", "id": "123"}], "Text"),
        ],
    )
    def test_extract_content(self, content, expected):
        """Test extracting content from various formats."""
        assert extract_text_content(content) == expected


class TestExtractPreviewMessage:
    """Test preview message extraction for session list display."""

    def test_basic_message(self):
        """Test extracting a basic user message."""
        payload = {"final_request": {"messages": [{"role": "user", "content": "Hello world"}]}}
        assert _extract_preview_message(payload) == "Hello world"

    def test_multiple_messages_returns_first_user(self):
        """Test that the first user message is returned (captures session intent)."""
        payload = {
            "final_request": {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "Answer"},
                    {"role": "user", "content": "Follow-up question"},
                ]
            }
        }
        assert _extract_preview_message(payload) == "First question"

    def test_truncates_long_messages(self):
        """Test that long messages are truncated to 100 chars."""
        long_message = "x" * 150
        payload = {"final_request": {"messages": [{"role": "user", "content": long_message}]}}
        result = _extract_preview_message(payload)
        assert result is not None
        assert len(result) == 103  # 100 chars + "..."
        assert result.endswith("...")

    def test_normalizes_whitespace(self):
        """Test that newlines and extra whitespace are collapsed."""
        payload = {"final_request": {"messages": [{"role": "user", "content": "Hello\n\nworld\n  test"}]}}
        assert _extract_preview_message(payload) == "Hello world test"

    def test_inline_system_reminder_matches_summary_preview(self):
        """Inline system reminders are stripped by the shared preview extractor."""
        payload = {
            "final_request": {
                "messages": [{"role": "user", "content": "real question <system-reminder>noise</system-reminder>"}]
            }
        }

        assert _extract_preview_message(payload) == "real question"

    def test_none_payload(self):
        """Test handling of None payload."""
        assert _extract_preview_message(None) is None

    def test_empty_payload(self):
        """Test handling of empty dict payload."""
        assert _extract_preview_message({}) is None

    def test_no_user_messages(self):
        """Test handling when no user messages present."""
        payload = {"final_request": {"messages": [{"role": "system", "content": "System prompt"}]}}
        assert _extract_preview_message(payload) is None

    def test_json_string_payload(self):
        """Test handling of JSON string payload from asyncpg."""
        import json

        payload_dict = {"final_request": {"messages": [{"role": "user", "content": "From JSON"}]}}
        payload_str = json.dumps(payload_dict)
        assert _extract_preview_message(payload_str) == "From JSON"

    def test_prefers_original_request_over_final_request(self):
        """Preview reflects what the user typed, not gateway-injected content.

        When inject_policy_awareness_anthropic (or any future injection) modifies
        the first user message in final_request, the preview must still show the
        user's original text — otherwise every session looks identical.
        """
        payload = {
            "original_request": {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
            "final_request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<policy-context>Your responses may be modified by the following active "
                            "policies before reaching the user: TestPolicy.</policy-context>\n\n"
                            "What is the capital of France?"
                        ),
                    }
                ]
            },
        }
        assert _extract_preview_message(payload) == "What is the capital of France?"

    def test_falls_back_to_final_request_when_original_missing(self):
        """Older payloads (recorded before original_request was stored) still produce a preview."""
        payload = {"final_request": {"messages": [{"role": "user", "content": "Legacy payload"}]}}
        assert _extract_preview_message(payload) == "Legacy payload"

    @pytest.mark.parametrize(
        "probe_content",
        ["count", "quota", "ping", "any-future-probe"],
    )
    def test_skips_probe_requests_with_max_tokens_1(self, probe_content):
        """Test that requests with max_tokens=1 are skipped regardless of content.

        Claude Code sends internal probes (token counting, quota checks) with
        max_tokens=1. This structural signal catches all probes without needing
        a content blocklist.
        """
        payload = {
            "final_request": {
                "max_tokens": 1,
                "messages": [{"role": "user", "content": probe_content}],
            }
        }
        assert _extract_preview_message(payload) is None

    def test_normal_max_tokens_not_skipped(self):
        """Test that requests with normal max_tokens are not skipped."""
        payload = {
            "final_request": {
                "max_tokens": 32000,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        }
        assert _extract_preview_message(payload) == "Hello"

    def test_max_tokens_string_1_skipped(self):
        """Test that max_tokens as string "1" is also caught (JSON parsing)."""
        payload = {
            "final_request": {
                "max_tokens": "1",
                "messages": [{"role": "user", "content": "quota"}],
            }
        }
        assert _extract_preview_message(payload) is None

    def test_missing_max_tokens_not_skipped(self):
        """Test that missing max_tokens doesn't skip (backwards compat)."""
        payload = {
            "final_request": {
                "messages": [{"role": "user", "content": "Hello"}],
            }
        }
        assert _extract_preview_message(payload) == "Hello"


class TestSafeParseJson:
    """Test safe JSON parsing."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ('{"key": "value"}', {"key": "value"}),
            ("{}", {}),
            ('{"nested": {"a": 1}}', {"nested": {"a": 1}}),
            ("invalid", None),
            ("[]", None),  # Not a dict
            ('"string"', None),  # Not a dict
        ],
    )
    def test_parse_json(self, input_str, expected):
        """Test JSON parsing with various inputs."""
        assert _safe_parse_json(input_str) == expected


class TestExtractToolCalls:
    """Test tool call extraction from messages."""

    def test_openai_style_tool_calls(self):
        """Test extracting OpenAI-style tool calls."""
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
                }
            ]
        }

        result = _extract_tool_calls(message)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_CALL
        assert result[0].tool_name == "read_file"
        assert result[0].tool_call_id == "call_123"
        assert result[0].tool_input == {"path": "/tmp/test"}

    def test_anthropic_style_content_blocks(self):
        """Test extracting Anthropic-style tool_use content blocks."""
        message = {
            "content": [
                {"type": "text", "text": "Let me read that file"},
                {"type": "tool_use", "id": "toolu_123", "name": "Read", "input": {"file": "test.py"}},
            ]
        }

        result = _extract_tool_calls(message)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_CALL
        assert result[0].tool_name == "Read"
        assert result[0].tool_call_id == "toolu_123"
        assert result[0].tool_input == {"file": "test.py"}

    def test_no_tool_calls(self):
        """Test message without tool calls."""
        message = {"content": "Hello world"}
        result = _extract_tool_calls(message)
        assert len(result) == 0

    def test_explicit_none_tool_calls(self):
        """Test message with tool_calls explicitly set to None.

        This case occurs in real OpenAI responses where tool_calls is present
        but null, not just missing from the dict.
        """
        message = {"content": "Hello world", "tool_calls": None}
        result = _extract_tool_calls(message)
        assert len(result) == 0


class TestAnthropicToolResultExtraction:
    """Test Anthropic-style tool_result extraction in user messages.

    When a user message contains a list with tool_result blocks, they are
    extracted into separate TOOL_RESULT messages with tool_call_id (from
    tool_use_id), and text blocks are extracted as separate USER messages.
    """

    def test_single_tool_result_in_user_message(self):
        """Test extracting a single tool_result from a user message."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "result text"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "result text"
        assert result[0].tool_call_id == "toolu_abc123"

    def test_mixed_text_and_tool_result(self):
        """Test extracting mixed text + tool_result blocks from a user message."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's what I found"},
                        {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "Tool output"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 2
        assert result[0].message_type == MessageType.USER
        assert result[0].content == "Here's what I found"
        assert result[1].message_type == MessageType.TOOL_RESULT
        assert result[1].content == "Tool output"
        assert result[1].tool_call_id == "toolu_xyz"

    def test_multiple_tool_result_blocks(self):
        """Test extracting multiple tool_result blocks from a user message."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "First result"},
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Second result"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 2
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "First result"
        assert result[0].tool_call_id == "toolu_1"
        assert result[1].message_type == MessageType.TOOL_RESULT
        assert result[1].content == "Second result"
        assert result[1].tool_call_id == "toolu_2"

    def test_tool_result_with_none_content(self):
        """Test that tool_result with None content becomes empty string."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_empty", "content": None},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == ""
        assert result[0].tool_call_id == "toolu_empty"

    def test_tool_result_with_string_content(self):
        """Test that tool_result with string content (not list) is extracted."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_str", "content": "Plain string result"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "Plain string result"
        assert result[0].tool_call_id == "toolu_str"

    def test_tool_result_missing_tool_use_id(self):
        """Test that tool_result without tool_use_id gets tool_call_id=None.

        The frontend cannot pair such results with tool calls, but parsing
        must not crash.
        """
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": "orphan result"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "orphan result"
        assert result[0].tool_call_id is None

    def test_tool_result_with_nested_content_blocks(self):
        """Test tool_result with nested content blocks (list of text blocks)."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_nested",
                            "content": [
                                {"type": "text", "text": "First part"},
                                {"type": "text", "text": "Second part"},
                            ],
                        },
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        # Nested content blocks should be concatenated with newlines
        assert result[0].content == "First part\nSecond part"
        assert result[0].tool_call_id == "toolu_nested"

    def test_text_only_user_message_not_special_path(self):
        """Test that text-only user message (no tool_results) uses normal path."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Just a question"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.USER
        assert result[0].content == "Just a question"
        assert result[0].tool_call_id is None

    def test_whitespace_only_text_blocks_filtered(self):
        """Test that whitespace-only text blocks are filtered out."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "   \n\t  "},  # Whitespace only
                        {"type": "tool_result", "tool_use_id": "toolu_ws", "content": "Result"},
                        {"type": "text", "text": "Valid text"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 2
        # Whitespace-only text should be filtered
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "Result"
        assert result[1].message_type == MessageType.USER
        assert result[1].content == "Valid text"

    def test_tool_result_with_is_error_true(self):
        """Test that tool_result with is_error=True is propagated."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_error",
                            "content": "Error occurred",
                            "is_error": True,
                        },
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].content == "Error occurred"
        assert result[0].tool_call_id == "toolu_error"
        assert result[0].is_error is True

    def test_tool_result_with_is_error_false(self):
        """Test that is_error=False is normalized to None (no error badge)."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_ok",
                            "content": "Success",
                            "is_error": False,
                        },
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].is_error is None

    def test_complex_mixed_blocks(self):
        """Test a complex scenario with multiple text + tool_result blocks."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First observation"},
                        {"type": "tool_result", "tool_use_id": "tool1", "content": "Data from tool 1"},
                        {"type": "text", "text": "Second observation"},
                        {"type": "tool_result", "tool_use_id": "tool2", "content": "Data from tool 2"},
                    ],
                }
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 4
        assert result[0].message_type == MessageType.USER
        assert result[0].content == "First observation"
        assert result[1].message_type == MessageType.TOOL_RESULT
        assert result[1].tool_call_id == "tool1"
        assert result[2].message_type == MessageType.USER
        assert result[2].content == "Second observation"
        assert result[3].message_type == MessageType.TOOL_RESULT
        assert result[3].tool_call_id == "tool2"


class TestParseRequestMessages:
    """Test request message parsing."""

    def test_simple_messages(self):
        """Test parsing simple text messages."""
        request = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 2
        assert result[0].message_type == MessageType.SYSTEM
        assert result[0].content == "You are helpful"
        assert result[1].message_type == MessageType.USER
        assert result[1].content == "Hello"

    def test_unrecognized_role_raises_error(self):
        """Test that unrecognized message roles raise ValueError."""
        request = {
            "messages": [
                {"role": "unknown_role", "content": "Hello"},
            ]
        }

        with pytest.raises(ValueError, match="Unrecognized message role: 'unknown_role'"):
            _parse_request_messages(request)

    def test_assistant_message_with_tool_calls(self):
        """Test parsing assistant messages with tool_calls in request.

        When conversation history includes an assistant message that made
        tool calls, those tool calls must be extracted and included.
        """
        request = {
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": '{"temperature": 22, "conditions": "sunny"}',
                },
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 3
        # First: user message
        assert result[0].message_type == MessageType.USER
        assert "weather" in result[0].content.lower()
        # Second: tool call from assistant
        assert result[1].message_type == MessageType.TOOL_CALL
        assert result[1].tool_name == "get_weather"
        assert result[1].tool_call_id == "call_abc123"
        # Third: tool result
        assert result[2].message_type == MessageType.TOOL_RESULT
        assert result[2].tool_call_id == "call_abc123"

    def test_tool_result_message(self):
        """Test parsing tool result messages."""
        request = {
            "messages": [
                {"role": "tool", "content": "File contents...", "tool_call_id": "call_123"},
            ]
        }

        result = _parse_request_messages(request)

        assert len(result) == 1
        assert result[0].message_type == MessageType.TOOL_RESULT
        assert result[0].tool_call_id == "call_123"


class TestParseResponseMessages:
    """Test response message parsing."""

    def test_simple_response(self):
        """Test parsing simple text response."""
        response = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}

        result = _parse_response_messages(response)

        assert len(result) == 1
        assert result[0].message_type == MessageType.ASSISTANT
        assert result[0].content == "Hello!"

    def test_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check",
                        "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}],
                    }
                }
            ]
        }

        result = _parse_response_messages(response)

        assert len(result) == 2
        assert result[0].message_type == MessageType.ASSISTANT
        assert result[1].message_type == MessageType.TOOL_CALL

    def test_anthropic_text_response(self):
        """Test parsing Anthropic-format response with text content."""
        response = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": DEFAULT_TEST_MODEL,
            "stop_reason": "end_turn",
        }

        result = _parse_response_messages(response)

        assert len(result) == 1
        assert result[0].message_type == MessageType.ASSISTANT
        assert result[0].content == "Hello from Claude!"

    def test_anthropic_response_with_tool_use(self):
        """Test parsing Anthropic-format response with tool_use content blocks."""
        response = {
            "id": "msg_test456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that file."},
                {"type": "tool_use", "id": "toolu_123", "name": "read_file", "input": {"path": "/foo.py"}},
            ],
            "model": DEFAULT_TEST_MODEL,
            "stop_reason": "tool_use",
        }

        result = _parse_response_messages(response)

        assert len(result) == 2
        assert result[0].message_type == MessageType.ASSISTANT
        assert result[0].content == "Let me read that file."
        assert result[1].message_type == MessageType.TOOL_CALL
        assert result[1].tool_name == "read_file"
        assert result[1].tool_call_id == "toolu_123"

    def test_anthropic_empty_text_response(self):
        """Test parsing Anthropic response with empty content blocks."""
        response = {
            "id": "msg_test789",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": DEFAULT_TEST_MODEL,
            "stop_reason": "end_turn",
        }

        result = _parse_response_messages(response)

        assert len(result) == 0


class TestBuildTurn:
    """Test building conversation turns from events."""

    def test_simple_turn(self):
        """Test building a simple request/response turn."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {"messages": [{"role": "user", "content": "Hello"}]},
                    "final_request": {"messages": [{"role": "user", "content": "Hello"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "event_type": "transaction.streaming_response_recorded",
                "payload": {
                    "original_response": {"choices": [{"message": {"content": "Hi!"}}]},
                    "final_response": {"choices": [{"message": {"content": "Hi!"}}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 1),
            },
        ]

        turn = _build_turn("call-123", events)

        assert turn.call_id == "call-123"
        assert turn.model == "gpt-4"
        assert len(turn.request_messages) == 1
        assert len(turn.response_messages) == 1
        assert not turn.had_policy_intervention

    def test_request_params_allowlist(self):
        """Test that request_params only includes allowlisted fields."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1024,
                        "model": "gpt-4",
                        "stream": True,
                        "metadata": {"api_key": "secret"},
                        "system": "You are helpful",
                        "tools": [{"name": "tool1"}, {"name": "tool2"}],
                    },
                    "final_request": {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1024,
                        "model": "gpt-4",
                        "stream": True,
                        "metadata": {"api_key": "secret"},
                        "system": "You are helpful",
                        "tools": [{"name": "tool1"}, {"name": "tool2"}],
                    },
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
        ]

        turn = _build_turn("call-123", events)

        assert turn.request_params is not None
        assert turn.request_params["max_tokens"] == 1024
        assert turn.request_params["model"] == "gpt-4"
        assert turn.request_params["stream"] is True
        assert turn.request_params["tools_count"] == 2
        # Sensitive/unknown fields must NOT leak
        assert "metadata" not in turn.request_params
        assert "system" not in turn.request_params
        assert "messages" not in turn.request_params
        assert "tools" not in turn.request_params

    def test_turn_with_policy_intervention(self):
        """Test turn with policy modification."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {"messages": [{"role": "user", "content": "Original"}]},
                    "final_request": {"messages": [{"role": "user", "content": "Modified"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "event_type": "policy.judge.tool_call_blocked",
                "payload": {"summary": "Tool call blocked for safety"},
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
        ]

        turn = _build_turn("call-123", events)

        assert turn.had_policy_intervention
        assert turn.request_was_modified
        assert turn.original_request_messages is not None
        assert turn.original_request_messages[0].content == "Original"
        assert len(turn.annotations) == 1
        assert turn.annotations[0].policy_name == "judge"

    def test_missing_final_request_raises_error(self):
        """Test that missing final_request raises KeyError."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {"messages": [{"role": "user", "content": "Hello"}]},
                    # final_request is missing
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
        ]

        with pytest.raises(KeyError, match="final_request"):
            _build_turn("call-123", events)

    def test_missing_final_response_raises_error(self):
        """Test that missing final_response raises KeyError."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {"messages": [{"role": "user", "content": "Hello"}]},
                    "final_request": {"messages": [{"role": "user", "content": "Hello"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "event_type": "transaction.streaming_response_recorded",
                "payload": {
                    "original_response": {"choices": [{"message": {"content": "Hi!"}}]},
                    # final_response is missing
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 1),
            },
        ]

        with pytest.raises(KeyError, match="final_response"):
            _build_turn("call-123", events)

    def test_anthropic_turn_with_text_response(self):
        """Test building a turn from Anthropic-format request and response events."""
        events: list[StoredEvent] = [
            {
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": DEFAULT_TEST_MODEL,
                    "original_request": {
                        "model": DEFAULT_TEST_MODEL,
                        "messages": [{"role": "user", "content": "Hello Claude"}],
                        "max_tokens": 1024,
                    },
                    "final_request": {
                        "model": DEFAULT_TEST_MODEL,
                        "messages": [{"role": "user", "content": "Hello Claude"}],
                        "max_tokens": 1024,
                    },
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "event_type": "transaction.non_streaming_response_recorded",
                "payload": {
                    "original_response": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello! How can I help?"}],
                        "model": DEFAULT_TEST_MODEL,
                        "stop_reason": "end_turn",
                    },
                    "final_response": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello! How can I help?"}],
                        "model": DEFAULT_TEST_MODEL,
                        "stop_reason": "end_turn",
                    },
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 1),
            },
        ]

        turn = _build_turn("call-456", events)

        assert turn.call_id == "call-456"
        assert turn.model == DEFAULT_TEST_MODEL
        assert len(turn.request_messages) == 1
        assert turn.request_messages[0].content == "Hello Claude"
        assert len(turn.response_messages) == 1
        assert turn.response_messages[0].content == "Hello! How can I help?"
        assert turn.response_messages[0].message_type == MessageType.ASSISTANT
        assert not turn.had_policy_intervention


class TestFetchSessionList:
    """Test fetching session list from database."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful session list fetching."""
        mock_rows = [
            {
                "session_id": "session-1",
                "first_ts": datetime(2025, 1, 15, 10, 0, 0),
                "last_ts": datetime(2025, 1, 15, 11, 0, 0),
                "total_events": 10,
                "turn_count": 3,
                "policy_interventions": 1,
                "models_used": "gpt-4,claude-3",
                "preview_message": "Hello world",
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1  # Total count
        # First fetch() = main session aggregation; second = user_ids lookup.
        mock_conn.fetch.side_effect = [mock_rows, []]

        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool)

        assert result.total == 1
        assert result.offset == 0
        assert result.has_more is False
        assert len(result.sessions) == 1
        assert result.sessions[0].session_id == "session-1"
        assert result.sessions[0].turn_count == 3
        assert result.sessions[0].policy_interventions == 1
        assert "gpt-4" in result.sessions[0].models_used
        assert result.sessions[0].preview_message == "Hello world"
        assert result.sessions[0].user_ids == []

    @pytest.mark.asyncio
    async def test_fetch_with_offset(self):
        """Test fetching with offset for pagination."""
        mock_rows = [
            {
                "session_id": "session-2",
                "first_ts": datetime(2025, 1, 15, 10, 0, 0),
                "last_ts": datetime(2025, 1, 15, 11, 0, 0),
                "total_events": 5,
                "turn_count": 2,
                "policy_interventions": 0,
                "models_used": "gpt-4",
                "preview_message": None,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 100  # Total count
        mock_conn.fetch.side_effect = [mock_rows, []]

        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool, offset=50)

        assert result.total == 100
        assert result.offset == 50
        assert result.has_more is True  # 50 + 1 < 100
        assert len(result.sessions) == 1
        assert result.sessions[0].preview_message is None

    @pytest.mark.asyncio
    async def test_empty_result(self):
        """Test when no sessions found."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 0  # Total count
        mock_conn.fetch.return_value = []

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool)

        assert result.total == 0
        assert result.offset == 0
        assert result.has_more is False
        assert result.sessions == []

    @pytest.mark.asyncio
    async def test_unfiltered_pg_list_uses_session_summaries_without_payload(self):
        """Unfiltered list hot path reads session_summaries and never selects full payloads."""
        summary_rows = [
            {
                "session_id": "session-1",
                "first_ts": datetime(2025, 1, 15, 10, 0, 0),
                "last_ts": datetime(2025, 1, 15, 11, 0, 0),
                "total_events": 10,
                "turn_count": 3,
                "policy_interventions": 1,
                "models_used": "gpt-4,claude-3",
                "preview_message": "Hello world",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.side_effect = [summary_rows, []]
        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool)

        queries = [call.args[0].lower() for call in mock_conn.fetch.call_args_list]
        assert result.sessions[0].preview_message == "Hello world"
        assert result.sessions[0].models_used == ["claude-3", "gpt-4"]
        assert "from session_summaries" in queries[0]
        assert "payload" not in queries[0]
        assert "request_payload" not in queries[0]

    @pytest.mark.asyncio
    async def test_pg_summary_null_preview_without_payload_returns_none(self):
        """PG summary rows with NULL preview return None without payload fallback."""
        summary_rows = [
            {
                "session_id": "session-null-preview",
                "first_ts": datetime(2025, 1, 15, 10, 0, 0),
                "last_ts": datetime(2025, 1, 15, 11, 0, 0),
                "total_events": 1,
                "turn_count": 1,
                "policy_interventions": 0,
                "models_used": "gpt-4",
                "preview_message": None,
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.side_effect = [summary_rows, []]
        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool)

        assert result.sessions[0].preview_message is None

    @pytest.mark.asyncio
    async def test_pg_summary_models_are_sorted_like_aggregation(self):
        """Summary models are sorted to match existing list output."""
        summary_rows = [
            {
                "session_id": "session-model-order",
                "first_ts": datetime(2025, 1, 15, 10, 0, 0),
                "last_ts": datetime(2025, 1, 15, 11, 0, 0),
                "total_events": 2,
                "turn_count": 2,
                "policy_interventions": 0,
                "models_used": "z-model,a-model",
                "preview_message": "Hello",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.side_effect = [summary_rows, []]
        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_list(limit=10, db_pool=mock_pool)

        assert result.sessions[0].models_used == ["a-model", "z-model"]

    @pytest.mark.asyncio
    async def test_pg_search_path_still_uses_existing_aggregation(self):
        """Full-text search remains on conversation_events aggregation."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 0
        mock_conn.fetch.return_value = []
        mock_pool = MagicMock()
        mock_pool.is_sqlite = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        await fetch_session_list(limit=10, db_pool=mock_pool, search=SessionSearchParams(model="gpt-4"))

        query = mock_conn.fetch.call_args.args[0].lower()
        assert "from conversation_events" in query
        assert "request_payload" in query


class TestFetchSessionDetail:
    """Test fetching session detail from database."""

    @pytest.mark.asyncio
    async def test_iter_session_turns_emits_request_message_deltas_with_preflight_excluded_from_count(self):
        """Detail turns carry display-equivalent request deltas instead of cumulative history."""
        ranges = [
            CallEventRange(
                call_id="call-1",
                first_ts=datetime(2025, 1, 15, 10, 0, 0),
                last_ts=datetime(2025, 1, 15, 10, 0, 0),
            ),
            CallEventRange(
                call_id="preflight",
                first_ts=datetime(2025, 1, 15, 10, 0, 30),
                last_ts=datetime(2025, 1, 15, 10, 0, 30),
            ),
            CallEventRange(
                call_id="call-2",
                first_ts=datetime(2025, 1, 15, 10, 1, 0),
                last_ts=datetime(2025, 1, 15, 10, 1, 0),
            ),
        ]
        rows_by_call = {
            "call-1": [
                {
                    "call_id": "call-1",
                    "event_type": "transaction.request_recorded",
                    "payload": {"final_request": {"messages": [{"role": "user", "content": "Hi"}]}},
                    "created_at": ranges[0].first_ts,
                }
            ],
            "preflight": [
                {
                    "call_id": "preflight",
                    "event_type": "transaction.request_recorded",
                    "payload": {
                        "final_request": {
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "quota probe"}],
                        }
                    },
                    "created_at": ranges[1].first_ts,
                }
            ],
            "call-2": [
                {
                    "call_id": "call-2",
                    "event_type": "transaction.request_recorded",
                    "payload": {
                        "original_request": {
                            "messages": [
                                {"role": "user", "content": "Hi"},
                                {"role": "assistant", "content": "Hello!"},
                                {"role": "user", "content": "Use forbidden tool"},
                            ]
                        },
                        "final_request": {
                            "messages": [
                                {"role": "user", "content": "Hi"},
                                {"role": "assistant", "content": "Hello!"},
                                {"role": "user", "content": "Use safe tool"},
                            ]
                        },
                    },
                    "created_at": ranges[2].first_ts,
                }
            ],
        }
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)

        def fetch_rows(_query, _session_id, *args):
            call_ids = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
            return [row for call_id in call_ids for row in rows_by_call[call_id]]

        mock_conn.fetch.side_effect = fetch_rows
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        turns = [turn async for turn in iter_session_turns("session-1", mock_pool, ranges)]

        request_contents = [[message.content for message in turn.request_messages] for turn in turns]
        original_contents = [message.content for message in turns[2].original_request_messages or []]
        final_full_contents = [message.content for message in turns[2].request_messages_full or []]
        assert request_contents == [["Hi"], ["quota probe"], ["Hello!", "Use safe tool"]]
        assert original_contents == ["Hi", "Hello!", "Use forbidden tool"]
        assert final_full_contents == ["Hi", "Hello!", "Use safe tool"]
        assert request_contents[0] + request_contents[2] == final_full_contents

    @pytest.mark.asyncio
    async def test_iter_session_turns_fetches_payloads_in_batches(self):
        """Payload fetch query count grows by batch count, not by turn count."""
        ranges = [
            CallEventRange(
                call_id=f"call-{index}",
                first_ts=datetime(2025, 1, 15, 10, index, 0),
                last_ts=datetime(2025, 1, 15, 10, index, 1),
            )
            for index in range(51)
        ]
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)

        def fetch_rows(_query, _session_id, *args):
            call_ids = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
            return [
                {
                    "call_id": call_id,
                    "event_type": "transaction.request_recorded",
                    "payload": {"final_request": {"messages": [{"role": "user", "content": call_id}]}},
                    "created_at": ranges[int(call_id.split("-")[1])].first_ts,
                }
                for call_id in call_ids
            ]

        mock_conn.fetch.side_effect = fetch_rows
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        turns = [turn async for turn in iter_session_turns("session-1", mock_pool, ranges)]

        assert len(turns) == 51
        assert mock_conn.fetch.await_count == 3
        assert all("call_id = any($2)" in call.args[0].lower() for call in mock_conn.fetch.call_args_list)

    @pytest.mark.asyncio
    async def test_iter_session_turns_fetches_sqlite_payload_batches_with_expanded_in_clause(self):
        ranges = [
            CallEventRange(
                call_id="call-1",
                first_ts=datetime(2025, 1, 15, 10, 0, 0),
                last_ts=datetime(2025, 1, 15, 10, 0, 10),
            ),
            CallEventRange(
                call_id="call-2",
                first_ts=datetime(2025, 1, 15, 10, 1, 0),
                last_ts=datetime(2025, 1, 15, 10, 1, 10),
            ),
            CallEventRange(
                call_id="call-3",
                first_ts=datetime(2025, 1, 15, 10, 2, 0),
                last_ts=datetime(2025, 1, 15, 10, 2, 10),
            ),
        ]
        rows_by_call = {
            "call-1": [
                {
                    "call_id": "call-1",
                    "event_type": "transaction.request_recorded",
                    "payload": {"final_request": {"messages": [{"role": "user", "content": "first"}]}},
                    "created_at": ranges[0].first_ts,
                }
            ],
            "call-2": [
                {
                    "call_id": "call-2",
                    "event_type": "transaction.request_recorded",
                    "payload": {
                        "final_request": {
                            "messages": [
                                {"role": "user", "content": "first"},
                                {"role": "user", "content": "second"},
                            ]
                        }
                    },
                    "created_at": ranges[1].first_ts,
                },
                {
                    "call_id": "call-2",
                    "event_type": "transaction.request_recorded",
                    "payload": {
                        "final_request": {
                            "messages": [
                                {"role": "user", "content": "first"},
                                {"role": "user", "content": "late"},
                            ]
                        }
                    },
                    "created_at": datetime(2025, 1, 15, 10, 1, 30),
                },
            ],
            "call-3": [
                {
                    "call_id": "call-3",
                    "event_type": "transaction.request_recorded",
                    "payload": {
                        "final_request": {
                            "messages": [
                                {"role": "user", "content": "first"},
                                {"role": "user", "content": "second"},
                                {"role": "user", "content": "third"},
                            ]
                        }
                    },
                    "created_at": ranges[2].first_ts,
                }
            ],
        }
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)

        def fetch_rows(_query, _session_id, *call_ids):
            return [row for call_id in call_ids for row in rows_by_call[call_id]]

        mock_conn.fetch.side_effect = fetch_rows
        mock_pool = MagicMock()
        mock_pool.is_sqlite = True
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        turns = [turn async for turn in iter_session_turns("session-1", mock_pool, ranges)]

        query, session_id, *call_ids = mock_conn.fetch.await_args.args
        assert [turn.call_id for turn in turns] == ["call-1", "call-2", "call-3"]
        assert [[message.content for message in turn.request_messages] for turn in turns] == [
            ["first"],
            ["second"],
            ["third"],
        ]
        assert "call_id IN ($2, $3, $4)" in query
        assert session_id == "session-1"
        assert call_ids == ["call-1", "call-2", "call-3"]

    @pytest.mark.asyncio
    async def test_iter_session_turns_bounds_reads_to_snapshot_last_timestamp(self):
        """Per-call streaming reads are bounded by the enumerated snapshot."""
        first_ts = datetime(2025, 1, 15, 10, 0, 0)
        snapshot_last = datetime(2025, 1, 15, 10, 1, 0)
        ranges = [CallEventRange(call_id="call-1", first_ts=first_ts, last_ts=snapshot_last)]
        rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {"final_request": {"messages": [{"role": "user", "content": "first"}]}},
                "created_at": first_ts,
            },
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {"final_request": {"messages": [{"role": "user", "content": "late"}]}},
                "created_at": datetime(2025, 1, 15, 10, 2, 0),
            },
        ]
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)
        mock_conn.fetch.return_value = rows
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        turns = [turn async for turn in iter_session_turns("session-1", mock_pool, ranges)]

        query, session_id, call_ids = mock_conn.fetch.await_args.args
        assert len(turns) == 1
        assert [message.content for message in turns[0].request_messages] == ["first"]
        assert "call_id = ANY($2)" in query
        assert session_id == "session-1"
        assert call_ids == ["call-1"]

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful session detail fetching."""
        mock_rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "original_request": {"messages": [{"role": "user", "content": "Hi"}]},
                    "final_request": {"messages": [{"role": "user", "content": "Hi"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "call_id": "call-1",
                "event_type": "transaction.streaming_response_recorded",
                "payload": {
                    "original_response": {"choices": [{"message": {"content": "Hello!"}}]},
                    "final_response": {"choices": [{"message": {"content": "Hello!"}}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 1),
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_detail("session-1", mock_pool)

        assert result.session_id == "session-1"
        assert len(result.turns) == 1
        assert result.turns[0].model == "gpt-4"

    @pytest.mark.asyncio
    async def test_fetch_session_detail_emits_multi_turn_request_message_deltas(self):
        rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_request": {"messages": [{"role": "user", "content": "Plan trip"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "call_id": "preflight",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_request": {
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "quota probe"}],
                    }
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 30),
            },
            {
                "call_id": "call-2",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_request": {
                        "messages": [
                            {"role": "user", "content": "Plan trip"},
                            {"role": "assistant", "content": "Where to?"},
                            {"role": "user", "content": "Lisbon"},
                        ]
                    }
                },
                "created_at": datetime(2025, 1, 15, 10, 1, 0),
            },
            {
                "call_id": "call-3",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_request": {
                        "messages": [
                            {"role": "user", "content": "Plan trip"},
                            {"role": "assistant", "content": "Where to?"},
                            {"role": "user", "content": "Lisbon"},
                            {"role": "assistant", "content": "Which dates?"},
                            {"role": "user", "content": "May"},
                        ]
                    }
                },
                "created_at": datetime(2025, 1, 15, 10, 2, 0),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = rows
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_detail("session-1", mock_pool)

        assert [turn.call_id for turn in result.turns] == ["call-1", "preflight", "call-2", "call-3"]
        assert [[message.content for message in turn.request_messages] for turn in result.turns] == [
            ["Plan trip"],
            ["quota probe"],
            ["Where to?", "Lisbon"],
            ["Which dates?", "May"],
        ]

    @pytest.mark.asyncio
    async def test_no_events_found(self):
        """Test error when no events found."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(ValueError, match="No events found"):
            await fetch_session_detail("nonexistent", mock_pool)

    @pytest.mark.asyncio
    async def test_unexpected_payload_type_raises_error(self):
        """Test that unexpected payload type raises TypeError."""
        mock_rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": 12345,  # Unexpected type (not dict or str)
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(TypeError, match="Unexpected payload type: int"):
            await fetch_session_detail("session-1", mock_pool)

    @pytest.mark.asyncio
    async def test_string_created_at_is_parsed(self):
        """Test that string created_at (from SQLite) is parsed into datetime."""
        mock_rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {"final_model": "gpt-4", "final_request": {"messages": []}},
                "created_at": "2025-01-15T10:00:00",
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        result = await fetch_session_detail("session-1", mock_pool)
        assert result.first_timestamp == "2025-01-15T10:00:00"

    @pytest.mark.asyncio
    async def test_unexpected_created_at_type_raises_error(self):
        """Test that unexpected created_at type raises TypeError."""
        mock_rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {"final_model": "gpt-4", "final_request": {"messages": []}},
                "created_at": 12345,  # Unexpected type (not datetime or str)
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(TypeError, match="got int"):
            await fetch_session_detail("session-1", mock_pool)

    @pytest.mark.asyncio
    async def test_streamed_session_detail_json_matches_existing_fetch_output(self):
        """Streaming detail JSON is semantically identical to existing SessionDetail."""
        from luthien_proxy.history import service

        rows = _equivalence_rows()
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)
        mock_conn.fetch.side_effect = [
            rows,
            [
                {"call_id": "call-1", "first_ts": rows[0]["created_at"], "last_ts": rows[1]["created_at"]},
                {"call_id": "call-2", "first_ts": rows[2]["created_at"], "last_ts": rows[4]["created_at"]},
            ],
            rows,
        ]
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        expected = await fetch_session_detail("session-1", mock_pool)

        body = await _collect_bytes(service.stream_session_detail_json("session-1", mock_pool))

        assert json.loads(body) == expected.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_streamed_detail_payload_fetches_are_scoped_to_one_call(self):
        """Streaming detail enumerates call ids without payload and fetches payloads per call."""
        from luthien_proxy.history import service

        rows = _equivalence_rows()
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)
        mock_conn.fetch.side_effect = [
            [
                {"call_id": "call-1", "first_ts": rows[0]["created_at"], "last_ts": rows[1]["created_at"]},
                {"call_id": "call-2", "first_ts": rows[2]["created_at"], "last_ts": rows[4]["created_at"]},
            ],
            rows[:2],
            rows[2:],
        ]
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn

        await _collect_bytes(service.stream_session_detail_json("session-1", mock_pool))

        queries = [call.args[0].lower() for call in mock_conn.fetch.call_args_list]
        assert "payload" not in queries[0]
        payload_queries = [query for query in queries[1:] if "payload" in query]
        assert len(payload_queries) == 1
        payload_call = mock_conn.fetch.call_args_list[1]
        assert payload_call.args[1] == "session-1"
        assert payload_call.args[2] == ["call-1", "call-2"]

    @pytest.mark.asyncio
    async def test_streamed_exports_match_existing_export_output(self):
        """Streaming markdown and JSONL exports equal existing exporters."""
        from luthien_proxy.history import service

        rows = _equivalence_rows()
        metadata = [
            {"call_id": "call-1", "first_ts": rows[0]["created_at"], "last_ts": rows[1]["created_at"]},
            {"call_id": "call-2", "first_ts": rows[2]["created_at"], "last_ts": rows[4]["created_at"]},
        ]
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)
        mock_conn.fetch.side_effect = [
            rows,
            metadata,
            rows,
            rows,
            metadata,
            rows,
        ]
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        expected = await fetch_session_detail("session-1", mock_pool)

        markdown = (await _collect_bytes(service.stream_session_markdown("session-1", mock_pool))).decode()
        jsonl = (await _collect_bytes(service.stream_session_jsonl("session-1", mock_pool))).decode()

        assert markdown == export_session_markdown(expected)
        assert jsonl == export_session_jsonl(expected)

    @pytest.mark.asyncio
    async def test_streamed_detail_uses_global_last_timestamp_for_interleaved_calls(self):
        """Streaming detail last timestamp matches global max event timestamp."""
        from luthien_proxy.history import service

        rows = [
            {
                "call_id": "call-1",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "gpt-4",
                    "final_request": {"messages": [{"role": "user", "content": "first"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 0, 0),
            },
            {
                "call_id": "call-2",
                "event_type": "transaction.request_recorded",
                "payload": {
                    "final_model": "claude-3",
                    "final_request": {"messages": [{"role": "user", "content": "second"}]},
                },
                "created_at": datetime(2025, 1, 15, 10, 1, 0),
            },
            {
                "call_id": "call-2",
                "event_type": "transaction.streaming_response_recorded",
                "payload": {"final_response": {"choices": [{"message": {"content": "second done"}}]}},
                "created_at": datetime(2025, 1, 15, 10, 2, 0),
            },
            {
                "call_id": "call-1",
                "event_type": "transaction.streaming_response_recorded",
                "payload": {"final_response": {"choices": [{"message": {"content": "first done"}}]}},
                "created_at": datetime(2025, 1, 15, 10, 3, 0),
            },
        ]
        ranges = [
            {"call_id": "call-1", "first_ts": rows[0]["created_at"], "last_ts": rows[3]["created_at"]},
            {"call_id": "call-2", "first_ts": rows[1]["created_at"], "last_ts": rows[2]["created_at"]},
        ]
        mock_conn = AsyncMock()
        _enable_mock_transaction(mock_conn)
        mock_conn.fetch.side_effect = [
            rows,
            ranges,
            rows,
            ranges,
            rows,
            rows,
        ]
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        expected = await fetch_session_detail("session-1", mock_pool)

        detail = json.loads(await _collect_bytes(service.stream_session_detail_json("session-1", mock_pool)))
        markdown = (await _collect_bytes(service.stream_session_markdown("session-1", mock_pool))).decode()

        assert detail["last_timestamp"] == expected.last_timestamp == "2025-01-15T10:03:00"
        assert "**Ended:** 2025-01-15T10:03:00" in markdown


class TestExportSessionMarkdown:
    """Test markdown export functionality."""

    def test_basic_export(self):
        """Test basic markdown export."""
        session = SessionDetail(
            session_id="test-session",
            first_timestamp="2025-01-15T10:00:00",
            last_timestamp="2025-01-15T11:00:00",
            turns=[
                ConversationTurn(
                    call_id="call-1",
                    timestamp="2025-01-15T10:00:00",
                    model="gpt-4",
                    request_messages=[ConversationMessage(message_type=MessageType.USER, content="Hello")],
                    response_messages=[ConversationMessage(message_type=MessageType.ASSISTANT, content="Hi there!")],
                    annotations=[],
                    had_policy_intervention=False,
                )
            ],
            total_policy_interventions=0,
            models_used=["gpt-4"],
        )

        markdown = export_session_markdown(session)

        assert "# Conversation History: test-session" in markdown
        assert "## Turn 1" in markdown
        assert "### User" in markdown
        assert "Hello" in markdown
        assert "### Assistant" in markdown
        assert "Hi there!" in markdown

    def test_export_with_tool_call(self):
        """Test markdown export with tool calls."""
        session = SessionDetail(
            session_id="test-session",
            first_timestamp="2025-01-15T10:00:00",
            last_timestamp="2025-01-15T11:00:00",
            turns=[
                ConversationTurn(
                    call_id="call-1",
                    timestamp="2025-01-15T10:00:00",
                    model="gpt-4",
                    request_messages=[],
                    response_messages=[
                        ConversationMessage(
                            message_type=MessageType.TOOL_CALL,
                            content="{}",
                            tool_name="read_file",
                            tool_input={"path": "/tmp/test"},
                        )
                    ],
                    annotations=[],
                    had_policy_intervention=False,
                )
            ],
            total_policy_interventions=0,
            models_used=["gpt-4"],
        )

        markdown = export_session_markdown(session)

        assert "### Tool Call" in markdown
        assert "`read_file`" in markdown
        assert '"/tmp/test"' in markdown

    def test_export_with_policy_annotations(self):
        """Test markdown export with policy annotations."""
        session = SessionDetail(
            session_id="test-session",
            first_timestamp="2025-01-15T10:00:00",
            last_timestamp="2025-01-15T11:00:00",
            turns=[
                ConversationTurn(
                    call_id="call-1",
                    timestamp="2025-01-15T10:00:00",
                    model="gpt-4",
                    request_messages=[],
                    response_messages=[],
                    annotations=[
                        PolicyAnnotation(
                            policy_name="judge",
                            event_type="policy.judge.tool_call_blocked",
                            summary="Dangerous operation blocked",
                        )
                    ],
                    had_policy_intervention=True,
                )
            ],
            total_policy_interventions=1,
            models_used=["gpt-4"],
        )

        markdown = export_session_markdown(session)

        assert "### Policy Annotations" in markdown
        assert "**judge**" in markdown
        assert "Dangerous operation blocked" in markdown
        assert "**Policy Interventions:** 1" in markdown


class TestExportSessionJsonl:
    def test_exports_turns_as_jsonl(self):
        session = SessionDetail(
            session_id="sess-1",
            first_timestamp="2026-03-31T10:00:00",
            last_timestamp="2026-03-31T10:01:00",
            turns=[
                ConversationTurn(
                    call_id="call-1",
                    timestamp="2026-03-31T10:00:00",
                    model="claude-3-opus",
                    request_messages=[
                        ConversationMessage(message_type=MessageType.USER, content="Hello"),
                    ],
                    response_messages=[
                        ConversationMessage(message_type=MessageType.ASSISTANT, content="Hi"),
                    ],
                    annotations=[],
                ),
                ConversationTurn(
                    call_id="call-2",
                    timestamp="2026-03-31T10:00:30",
                    model="claude-3-opus",
                    request_messages=[
                        ConversationMessage(message_type=MessageType.USER, content="Help"),
                    ],
                    response_messages=[
                        ConversationMessage(
                            message_type=MessageType.TOOL_CALL,
                            content="{}",
                            tool_name="read_file",
                            tool_call_id="tc-1",
                            tool_input={"path": "/tmp/x"},
                        ),
                    ],
                    annotations=[],
                ),
            ],
            total_policy_interventions=0,
            models_used=["claude-3-opus"],
        )

        result = export_session_jsonl(session)
        lines = result.strip().split("\n")
        assert len(lines) == 2

        line1 = json.loads(lines[0])
        assert line1["call_id"] == "call-1"
        assert line1["session_id"] == "sess-1"
        assert line1["model"] == "claude-3-opus"
        assert len(line1["request_messages"]) == 1
        assert len(line1["response_messages"]) == 1

        line2 = json.loads(lines[1])
        assert line2["call_id"] == "call-2"
        assert line2["response_messages"][0]["tool_name"] == "read_file"

    def test_empty_session_returns_empty_string(self):
        session = SessionDetail(
            session_id="sess-empty",
            first_timestamp="2026-03-31T10:00:00",
            last_timestamp="2026-03-31T10:00:00",
            turns=[],
            total_policy_interventions=0,
            models_used=[],
        )
        result = export_session_jsonl(session)
        assert result == ""

    def test_includes_original_messages_when_modified(self):
        session = SessionDetail(
            session_id="sess-mod",
            first_timestamp="2026-03-31T10:00:00",
            last_timestamp="2026-03-31T10:01:00",
            turns=[
                ConversationTurn(
                    call_id="call-mod",
                    timestamp="2026-03-31T10:00:00",
                    model="claude-3-opus",
                    request_messages=[
                        ConversationMessage(message_type=MessageType.USER, content="Hello"),
                    ],
                    response_messages=[
                        ConversationMessage(message_type=MessageType.ASSISTANT, content="[Link]\n\nHi"),
                    ],
                    original_response_messages=[
                        ConversationMessage(message_type=MessageType.ASSISTANT, content="Hi"),
                    ],
                    annotations=[],
                    had_policy_intervention=True,
                    response_was_modified=True,
                ),
            ],
            total_policy_interventions=1,
            models_used=["claude-3-opus"],
        )

        result = export_session_jsonl(session)
        record = json.loads(result)

        assert record["response_was_modified"] is True
        assert record["request_was_modified"] is False
        assert record["original_response_messages"][0]["content"] == "Hi"
        assert "original_request_messages" not in record
