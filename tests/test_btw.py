"""Tests for the /btw side-question logic (limbo.btw)."""

from __future__ import annotations

from limbo.btw import BTW_INSTRUCTION, build_btw_messages
from limbo.models import Message


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


