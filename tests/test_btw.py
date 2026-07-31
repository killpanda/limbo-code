"""Tests for the /btw side-question logic (limbo.btw)."""

from __future__ import annotations

import pytest

from limbo.btw import BTW_INSTRUCTION, btw_query, build_btw_messages
from limbo.models import CompletionMeta, Message, TextChunk, ThinkingChunk


def sample_history() -> list[Message]:
    return [
        Message(role="system", content="You are Limbo."),
        Message(role="user", content="fix the bug"),
        Message(role="assistant", content="working on it"),
    ]


class FakeLLMClient:
    def __init__(self, events):
        self.events = events
        self.calls: list[tuple[list[Message], list]] = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((messages, tools))
        for event in self.events:
            yield event


def test_build_btw_messages_keeps_history_byte_identical_for_prefix_cache():
    history = sample_history()
    messages = build_btw_messages(history, "what file was that?")

    # System prompt and history are reused untouched (same objects), so the
    # request's prefix matches the main conversation's cache exactly.
    assert messages[:-1] == history
    for original, reused in zip(history, messages):
        assert reused is original
    # The instruction and question travel in the trailing user message.
    assert messages[-1].role == "user"
    assert BTW_INSTRUCTION in messages[-1].content
    assert messages[-1].content.endswith("what file was that?")


def test_build_btw_messages_does_not_mutate_history():
    history = sample_history()
    build_btw_messages(history, "q")

    assert len(history) == 3
    assert history[0].content == "You are Limbo."


def test_build_btw_messages_without_system_message():
    history = [Message(role="user", content="hi")]
    messages = build_btw_messages(history, "q")

    assert [m.role for m in messages] == ["user", "user"]
    assert BTW_INSTRUCTION in messages[-1].content


@pytest.mark.asyncio
async def test_btw_query_streams_text_chunks_without_tools():
    client = FakeLLMClient(
        [
            ThinkingChunk(text="hmm"),
            TextChunk(text="the file is "),
            TextChunk(text="config.py"),
            CompletionMeta(usage={"total_tokens": 42}),
        ]
    )

    chunks = [c async for c in btw_query(client, sample_history(), "which file?")]

    assert chunks == ["the file is ", "config.py"]
    messages, tools = client.calls[0]
    assert tools == []
    assert "which file?" in messages[-1].content
    assert BTW_INSTRUCTION in messages[-1].content
    assert messages[0].content == "You are Limbo."


@pytest.mark.asyncio
async def test_btw_query_propagates_client_errors():
    class FailingClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="boom"):
        [c async for c in btw_query(FailingClient(), sample_history(), "q")]
