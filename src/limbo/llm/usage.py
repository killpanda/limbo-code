"""Token accounting: usage normalization and prompt-size estimation.

One module owns every token-counting rule in Limbo, so the agent loop and
compaction never touch a provider's usage field names:

- :func:`normalize_usage` turns a provider-specific ``usage`` dict (OpenAI
  chat completions, Anthropic messages, OpenAI Responses, DeepSeek cache
  counters) into a :class:`UsageTotals`.
- :func:`estimate_tokens` is the chars/4 heuristic used when no real usage
  figures exist (compaction trigger, split-point selection).
- :class:`PromptSizeEstimator` predicts the *next* request's prompt size
  from the last real figure plus the estimated cost of everything appended
  since (the watermark).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from limbo.models import Message


@dataclass(frozen=True)
class UsageTotals:
    """Provider-neutral token counters for one completion.

    ``None`` means the provider reported nothing usable for that counter.
    """

    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


def normalize_usage(usage: dict[str, Any] | None) -> UsageTotals:
    """Normalize provider-specific usage counters.

    Prompt: OpenAI reports ``prompt_tokens``; Anthropic reports
    ``input_tokens`` (which excludes cache reads/creation, so those are
    added back).

    Cached: DeepSeek reports ``prompt_cache_hit_tokens``, OpenAI nests
    ``prompt_tokens_details.cached_tokens``, the Responses API nests
    ``input_tokens_details.cached_tokens``, Anthropic reports
    ``cache_read_input_tokens``.

    Total: OpenAI-compatible providers report ``total_tokens`` (its
    ``prompt_tokens`` already includes cached tokens). Anthropic reports
    separate ``input_tokens`` / ``output_tokens``, with cache traffic
    reported apart — cache reads/creations are real processed (and billed)
    tokens, so they are added back; otherwise the cumulative counter
    massively undercounts cache-heavy sessions.
    """
    if not usage:
        return UsageTotals()

    prompt = usage.get("prompt_tokens")
    if not isinstance(prompt, int):
        input_tokens = usage.get("input_tokens")
        if isinstance(input_tokens, int):
            prompt = input_tokens + _cache_traffic(usage)
        else:
            prompt = None

    cached = usage.get("prompt_cache_hit_tokens")
    if not isinstance(cached, int):
        cached = None
        for key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(key)
            if isinstance(details, dict) and isinstance(
                details.get("cached_tokens"), int
            ):
                cached = details["cached_tokens"]
                break
        if cached is None:
            read = usage.get("cache_read_input_tokens")
            cached = read if isinstance(read, int) else None

    total = usage.get("total_tokens")
    if not isinstance(total, int):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) or isinstance(output_tokens, int):
            total = (input_tokens or 0) + (output_tokens or 0) + _cache_traffic(
                usage
            )
        else:
            total = None

    return UsageTotals(
        prompt_tokens=prompt, cached_tokens=cached, total_tokens=total
    )


def _cache_traffic(usage: dict[str, Any]) -> int:
    """Anthropic's separately-reported cache reads + creations."""
    result = 0
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        extra = usage.get(key)
        if isinstance(extra, int):
            result += extra
    return result


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate: chars / 4 plus a small per-message overhead.

    Conservative for Chinese text (~1.5–2 chars/token), so triggers may
    fire late on estimation alone; real usage figures are the primary path
    and compaction's ``reserve_tokens`` absorbs the slack. Used only for
    triggering and split-point selection, never recorded as measured usage.
    """
    total = 0
    for msg in messages:
        chars = len(msg.content or "") + len(msg.reasoning or "")
        if msg.tool_calls:
            chars += len(json.dumps(msg.tool_calls, ensure_ascii=False))
        total += chars // 4 + 4
    return total


class PromptSizeEstimator:
    """Predicts the size of the *next* request's prompt.

    With real usage: last prompt + estimated tokens of everything appended
    since (new user input, assistant messages, tool results). This catches
    the classic overflow case — a huge paste between turns — that a stale
    last-response figure alone would miss. Without usage (first iteration,
    post-compaction, or a provider that reports none): full estimate.
    """

    def __init__(self) -> None:
        self._last_prompt_tokens: int | None = None
        # len(messages) at the moment the real figure was recorded.
        self._watermark: int = 0

    def record(self, prompt_tokens: int, message_count: int) -> None:
        """Store a real prompt size, watermarking past ``message_count`` messages."""
        self._last_prompt_tokens = prompt_tokens
        self._watermark = message_count

    def reset(self, message_count: int) -> None:
        """Forget real figures after a history rewrite (e.g. compaction).

        Falls back to full estimation until the next response reports real
        prompt tokens; the watermark re-anchors so the stale slice of the
        old history is never double-counted.
        """
        self._last_prompt_tokens = None
        self._watermark = message_count

    def estimate_next(self, messages: list[Message]) -> int:
        """Estimated prompt size of the next request."""
        if self._last_prompt_tokens is None:
            return estimate_tokens(messages)
        return self._last_prompt_tokens + estimate_tokens(
            messages[self._watermark :]
        )
