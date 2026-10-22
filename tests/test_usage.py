"""Tests for the token-accounting module (usage normalization + estimation).

The agent loop and compaction consume this interface; neither knows any
provider's usage field names.
"""

from __future__ import annotations

from limbo.llm.usage import (
    PromptSizeEstimator,
    estimate_tokens,
)
from limbo.models import Message


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def _assistant(text: str | None, tool_calls: list[dict] | None = None) -> Message:
    return Message(role="assistant", content=text, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str = "read") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": {"path": "x"}},
    }


# -- normalize_usage: prompt tokens -------------------------------------------


# -- normalize_usage: cached tokens --------------------------------------------


# -- normalize_usage: total tokens ---------------------------------------------


# -- estimate_tokens -----------------------------------------------------------


# -- PromptSizeEstimator -------------------------------------------------------


def test_estimator_with_usage_adds_only_messages_since_watermark():
    before = [_user("x" * 400)]
    estimator = PromptSizeEstimator()
    estimator.record(prompt_tokens=10_000, message_count=len(before))
    # Nothing appended since the watermark: the real figure stands alone.
    assert estimator.estimate_next(before) == 10_000
    # A huge paste between turns is priced in incrementally.
    after = before + [_user("y" * 4000)]
    assert estimator.estimate_next(after) == 10_000 + estimate_tokens(after[1:])


