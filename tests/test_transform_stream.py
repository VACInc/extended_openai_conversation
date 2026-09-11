"""Tests for append-only normalization of chat completion streams."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
)
from custom_components.extended_openai_conversation.exceptions import (
    ParseArgumentsFailed,
    TokenLengthExceededError,
)

_KIND_RANK = {
    "thinking_content": 1,
    "content": 2,
    "tool_calls": 3,
}


def _chunk(
    *,
    content: Any = None,
    reasoning_content: str | None = None,
    reasoning: Any = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
    choices: list[Any] | None = None,
) -> SimpleNamespace:
    """Build a ChatCompletionChunk-like object."""
    if choices is not None:
        return SimpleNamespace(choices=choices, usage=usage)

    extra: dict[str, Any] = {}
    if reasoning_content is not None:
        extra["reasoning_content"] = reasoning_content
    if reasoning is not None:
        extra["reasoning"] = reasoning

    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
        model_extra=extra,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tool_call(
    *,
    index: int = 0,
    call_id: str = "call_1",
    name: str = "bash",
    arguments: str = '{"command": "true"}',
) -> SimpleNamespace:
    """Build a tool-call delta fragment."""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assert_append_only(deltas: list[dict[str, Any]]) -> None:
    """Fail if deltas would violate HA's per-message append-only contract."""
    current_kind = 0
    started = False
    for delta in deltas:
        if "role" in delta:
            assert delta["role"] == "assistant"
            current_kind = 0
            started = True
            continue
        kinds = [key for key in _KIND_RANK if key in delta]
        assert kinds, f"unexpected delta {delta}"
        assert started, f"payload before assistant role: {delta}"
        for key in kinds:
            rank = _KIND_RANK[key]
            assert rank >= current_kind, (
                f"append-only violation: {key} after kind {current_kind} in {delta}"
            )
            current_kind = rank


async def _collect(chunks: list[Any]) -> list[dict[str, Any]]:
    """Run _transform_stream against a fake async chunk stream."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.subentry = SimpleNamespace(data={})
    chat_log = MagicMock()

    async def fake_stream():
        for chunk in chunks:
            yield chunk

    return [delta async for delta in entity._transform_stream(chat_log, fake_stream())]


@pytest.mark.asyncio
async def test_plain_content_stream_is_append_only() -> None:
    """A normal content stream should still start one assistant message."""
    deltas = await _collect(
        [
            _chunk(content="The "),
            _chunk(content="kitchen light is on.", finish_reason="stop"),
        ]
    )

    _assert_append_only(deltas)
    assert [delta.get("role") for delta in deltas if "role" in delta] == ["assistant"]
    assert [delta["content"] for delta in deltas if "content" in delta] == [
        "The ",
        "kitchen light is on.",
    ]


@pytest.mark.asyncio
async def test_content_then_tool_call_then_content_is_append_only() -> None:
    """Content, a complete tool call, then more content must stay append-only."""
    deltas = await _collect(
        [
            _chunk(content="I'll run a command. "),
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="Done.", finish_reason="stop"),
        ]
    )

    _assert_append_only(deltas)
    contents = [delta["content"] for delta in deltas if "content" in delta]
    assert contents == ["I'll run a command. ", "Done."]
    tool_deltas = [delta for delta in deltas if "tool_calls" in delta]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["tool_calls"][0].tool_name == "bash"
    roles = [i for i, delta in enumerate(deltas) if delta.get("role") == "assistant"]
    assert len(roles) == 2
    assert roles[0] == 0


@pytest.mark.asyncio
async def test_tool_calls_finish_then_content_starts_new_message() -> None:
    """A tool_calls finish_reason must not leave later content on the same message."""
    deltas = await _collect(
        [
            _chunk(content="Calling the tool."),
            _chunk(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _chunk(content="Here is more speech."),
            _chunk(finish_reason="stop"),
        ]
    )

    _assert_append_only(deltas)
    role_indexes = [
        i for i, delta in enumerate(deltas) if delta.get("role") == "assistant"
    ]
    assert len(role_indexes) == 2
    first_tool = next(i for i, delta in enumerate(deltas) if "tool_calls" in delta)
    second_content = next(
        i
        for i, delta in enumerate(deltas)
        if delta.get("content") == "Here is more speech."
    )
    assert first_tool < role_indexes[1] < second_content


@pytest.mark.asyncio
async def test_reasoning_interleaved_with_content_is_append_only() -> None:
    """reasoning_content mixed with content must not go backwards in one message."""
    deltas = await _collect(
        [
            _chunk(content="Hello "),
            _chunk(reasoning_content="need to think"),
            _chunk(content="world.", finish_reason="stop"),
        ]
    )

    _assert_append_only(deltas)
    thinking = [
        delta["thinking_content"] for delta in deltas if "thinking_content" in delta
    ]
    assert thinking == ["need to think"]
    contents = [delta["content"] for delta in deltas if "content" in delta]
    assert contents == ["Hello ", "world."]
    assert sum(1 for delta in deltas if delta.get("role") == "assistant") == 2


@pytest.mark.asyncio
async def test_reasoning_field_before_content_stays_one_message() -> None:
    """A thinking prefix followed by content is already append-only."""
    deltas = await _collect(
        [
            _chunk(reasoning="plan the answer"),
            _chunk(content="The lights are on.", finish_reason="stop"),
        ]
    )

    _assert_append_only(deltas)
    assert [delta for delta in deltas if "role" in delta] == [{"role": "assistant"}]
    assert deltas[1]["thinking_content"] == "plan the answer"
    assert deltas[2]["content"] == "The lights are on."


@pytest.mark.asyncio
async def test_token_length_error_is_not_swallowed() -> None:
    """Real finish_reason=length errors must still raise."""
    with pytest.raises(TokenLengthExceededError):
        await _collect(
            [
                _chunk(content="partial"),
                _chunk(finish_reason="length"),
            ]
        )


@pytest.mark.asyncio
async def test_invalid_tool_arguments_still_raise() -> None:
    """Malformed tool-call JSON is a real error, not a stream glitch."""
    with pytest.raises(ParseArgumentsFailed):
        await _collect(
            [
                _chunk(
                    tool_calls=[_tool_call(arguments="{not-json")],
                    finish_reason="tool_calls",
                )
            ]
        )
