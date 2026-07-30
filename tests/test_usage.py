"""Tests for the token-accounting module (usage normalization + estimation).

The agent loop and compaction consume this interface; neither knows any
provider's usage field names.
"""

from __future__ import annotations

from limbo.llm.usage import (
    PromptSizeEstimator,
    estimate_tokens,
    normalize_usage,
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


def test_normalize_usage_openai_prompt_tokens():
    totals = normalize_usage({"prompt_tokens": 100, "total_tokens": 110})
    assert totals.prompt_tokens == 100


def test_normalize_usage_anthropic_prompt_includes_cache_traffic():
    # Anthropic's input_tokens excludes cache reads/creation; they are real
    # processed prompt tokens and must be added back.
    totals = normalize_usage(
        {
            "input_tokens": 871,
            "cache_read_input_tokens": 10240,
            "cache_creation_input_tokens": 0,
        }
    )
    assert totals.prompt_tokens == 871 + 10240


def test_normalize_usage_prompt_missing():
    assert normalize_usage(None).prompt_tokens is None
    assert normalize_usage({}).prompt_tokens is None
    assert normalize_usage({"total_tokens": 5}).prompt_tokens is None


# -- normalize_usage: cached tokens --------------------------------------------


def test_normalize_usage_cached_responses_dialect():
    # The Responses API nests cache hits under input_tokens_details.
    usage = {
        "input_tokens": 164,
        "output_tokens": 20,
        "total_tokens": 184,
        "input_tokens_details": {"cached_tokens": 64},
    }
    assert normalize_usage(usage).cached_tokens == 64


def test_normalize_usage_cached_existing_branches_win_first():
    assert normalize_usage({"prompt_cache_hit_tokens": 10}).cached_tokens == 10
    assert (
        normalize_usage({"prompt_tokens_details": {"cached_tokens": 20}}).cached_tokens
        == 20
    )
    assert normalize_usage({"cache_read_input_tokens": 30}).cached_tokens == 30
    assert normalize_usage({"input_tokens": 5}).cached_tokens is None
    assert normalize_usage(None).cached_tokens is None


# -- normalize_usage: total tokens ---------------------------------------------


def test_normalize_usage_total_openai_total_wins():
    # OpenAI's total_tokens already includes cached prompt tokens.
    totals = normalize_usage(
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    )
    assert totals.total_tokens == 110


def test_normalize_usage_total_anthropic_counts_cache_traffic():
    # Anthropic reports cache reads/creations apart from input_tokens; they
    # are real processed tokens and must count toward the session total.
    usage = {
        "input_tokens": 871,
        "output_tokens": 900,
        "cache_read_input_tokens": 10240,
        "cache_creation_input_tokens": 0,
    }
    assert normalize_usage(usage).total_tokens == 871 + 900 + 10240


def test_normalize_usage_total_anthropic_without_cache_keys():
    assert normalize_usage({"input_tokens": 5, "output_tokens": 7}).total_tokens == 12
    assert normalize_usage({"output_tokens": 3}).total_tokens == 3
    assert normalize_usage({}).total_tokens is None
    assert normalize_usage(None).total_tokens is None


# -- estimate_tokens -----------------------------------------------------------


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_counts_content_and_overhead():
    # 400 chars -> 100 tokens + 4 overhead.
    assert estimate_tokens([_user("x" * 400)]) == 104


def test_estimate_tokens_includes_tool_calls_json():
    msg = _assistant(None, tool_calls=[_tool_call("c1")])
    plain = _assistant(None)
    assert estimate_tokens([msg]) > estimate_tokens([plain])


# -- PromptSizeEstimator -------------------------------------------------------


def test_estimator_without_usage_falls_back_to_full_estimate():
    messages = [_user("x" * 400), _user("y" * 400)]
    estimator = PromptSizeEstimator()
    assert estimator.estimate_next(messages) == estimate_tokens(messages)


def test_estimator_with_usage_adds_only_messages_since_watermark():
    before = [_user("x" * 400)]
    estimator = PromptSizeEstimator()
    estimator.record(prompt_tokens=10_000, message_count=len(before))
    # Nothing appended since the watermark: the real figure stands alone.
    assert estimator.estimate_next(before) == 10_000
    # A huge paste between turns is priced in incrementally.
    after = before + [_user("y" * 4000)]
    assert estimator.estimate_next(after) == 10_000 + estimate_tokens(after[1:])


def test_estimator_reset_forgets_usage_and_reanchors():
    messages = [_user("x" * 400), _user("y" * 400)]
    estimator = PromptSizeEstimator()
    estimator.record(prompt_tokens=10_000, message_count=2)
    # History was rewritten (compaction): real figures predate the rewrite.
    estimator.reset(message_count=len(messages))
    assert estimator.estimate_next(messages) == estimate_tokens(messages)
