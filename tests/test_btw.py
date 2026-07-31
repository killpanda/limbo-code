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


def test_build_btw_messages_appends_instruction_to_system_prompt():
    history = sample_history()
    messages = build_btw_messages(history, "what file was that?")

    assert messages[0].role == "system"
    assert messages[0].content.startswith("You are Limbo.")
    assert BTW_INSTRUCTION in messages[0].content
    # History tail is reused as-is, question goes last.
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1].content == "what file was that?"


def test_build_btw_messages_does_not_mutate_history():
    history = sample_history()
    original_content = history[0].content
    build_btw_messages(history, "q")

    assert history[0].content == original_content
    assert len(history) == 3


def test_build_btw_messages_without_system_message():
    history = [Message(role="user", content="hi")]
    messages = build_btw_messages(history, "q")

    assert messages[0].role == "system"
    assert messages[0].content == BTW_INSTRUCTION
    assert [m.role for m in messages] == ["system", "user", "user"]


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
    assert messages[-1].content == "which file?"
    assert BTW_INSTRUCTION in messages[0].content


@pytest.mark.asyncio
async def test_btw_query_propagates_client_errors():
    class FailingClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="boom"):
        [c async for c in btw_query(FailingClient(), sample_history(), "q")]
