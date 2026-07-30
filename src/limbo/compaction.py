"""Context-window auto-compaction: pure decision and prompt-building logic.

No UI or agent dependencies (same layering as ``sessions.py``). The agent
loop consults :func:`should_compact` before each LLM call, and when it fires,
:func:`find_split_point` picks a safe cut, :func:`build_summary_prompt`
builds the summarization request, and :func:`make_summary_message` turns the
LLM's answer into a message that survives session repair. Token accounting
(normalization + estimation) lives in ``limbo.llm.usage``.

Design reference: RFC v2 on LIM-14 (pi-style compaction).
"""

from __future__ import annotations

from dataclasses import dataclass

from limbo.llm.usage import estimate_tokens
from limbo.models import Message

SUMMARY_TAG = "conversation-summary"
# Per-message serialization cap when rendering history into the summary
# prompt. Tool outputs dominate token volume; a summary never needs their
# full text, and an uncapped render could overflow the summary request.
SERIALIZE_CAP = 10_000

# The six sections every summary must carry (asserted by tests and used as
# the machine-checkable quality bar).
SUMMARY_SECTIONS = (
    "Goal",
    "Progress",
    "Key Decisions",
    "Files Touched",
    "Errors & Fixes",
    "Next Steps",
)


@dataclass(frozen=True)
class CompactionConfig:
    """Runtime knobs, mapped from the ``[compaction]`` config section."""

    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


def should_compact(
    estimated_prompt_tokens: int,
    context_window: int,
    cfg: CompactionConfig,
) -> bool:
    """Whether the *next* request is predicted to exceed the safe budget."""
    if not cfg.enabled:
        return False
    return estimated_prompt_tokens > context_window - cfg.reserve_tokens


def find_split_point(messages: list[Message], keep_recent_tokens: int) -> int | None:
    """Index where the kept region starts, or None if there is nothing to cut.

    Walks from the tail accumulating :func:`estimate_tokens` until
    ``keep_recent_tokens`` is exhausted. The boundary is then moved forward
    past any ``role="tool"`` messages so a kept region never starts with a
    dangling tool result (its assistant ``tool_calls`` message would have
    been summarized away) — the same invariant ``history.repair`` maintains.

    Returns None when the cut would leave nothing meaningful to summarize
    (boundary lands on or before the first non-system message), or when the
    boundary adjustment runs off the end. ``messages[0]`` (the system
    message) is never part of the summarized region.
    """
    total = 0
    idx = len(messages)
    while idx > 1:
        cost = estimate_tokens(messages[idx - 1 : idx])
        if total + cost > keep_recent_tokens:
            break
        total += cost
        idx -= 1
    # Never start the kept region with a dangling tool result.
    while idx < len(messages) and messages[idx].role == "tool":
        idx += 1
    if idx <= 1 or idx >= len(messages):
        return None
    return idx


def _serialize_message(msg: Message) -> str:
    parts: list[str] = []
    if msg.content:
        content = msg.content
        if len(content) > SERIALIZE_CAP:
            content = content[:SERIALIZE_CAP] + "\n[... truncated ...]"
        parts.append(content)
    if msg.tool_calls:
        calls = ", ".join(
            tc.get("function", {}).get("name", "?") for tc in msg.tool_calls
        )
        parts.append(f"[tool calls: {calls}]")
    if not parts:
        return ""
    return f"{msg.role}: " + "\n".join(parts)


def build_summary_prompt(
    to_summarize: list[Message],
    previous_summary: str | None = None,
) -> list[Message]:
    """Build the one-shot summarization request (a single user message).

    Iterative: when ``previous_summary`` exists it is included as established
    context, so the new summary folds the newly-dropped region into the old
    one instead of re-summarizing everything. Length is constrained by prompt
    (soft cap) — see RFC v2 §4.4 for why no hard max_tokens is used.
    """
    rendered = "\n\n".join(
        block for msg in to_summarize if (block := _serialize_message(msg))
    )
    sections = "\n".join(f"## {name}" for name in SUMMARY_SECTIONS)
    prompt = (
        "You are compacting a coding-agent conversation that is about to "
        "overflow its context window. Summarize the conversation excerpt "
        "below so work can continue without it.\n\n"
        "Requirements:\n"
        "- Write the summary with EXACTLY these six section headers:\n"
        f"{sections}\n"
        "- Keep the total under 2000 characters; be terse and factual.\n"
        "- Preserve concrete file paths, command names, error messages, "
        "and decisions verbatim where they matter.\n"
    )
    if previous_summary:
        prompt += (
            "\nAn earlier summary covers the conversation before this "
            "excerpt. Fold the excerpt into it: produce ONE updated summary "
            "in the same six-section format, not a separate addendum.\n\n"
            "<previous-summary>\n"
            f"{previous_summary}\n"
            "</previous-summary>\n"
        )
    prompt += (
        "\n<conversation>\n"
        f"{rendered}\n"
        "</conversation>\n"
    )
    return [Message(role="user", content=prompt)]


def make_summary_message(summary: str) -> Message:
    """Wrap an LLM-produced summary as a synthetic user message.

    User role (not system) so ``history.repair`` keeps it on resume — it
    drops only the leading system message.
    """
    return Message(
        role="user",
        content=(
            f"<{SUMMARY_TAG}>\n"
            "Earlier conversation was compacted into this summary. Treat it "
            "as established context; the messages after it are the most "
            "recent part of the conversation.\n\n"
            f"{summary.strip()}\n"
            f"</{SUMMARY_TAG}>"
        ),
    )


def is_summary_message(msg: Message) -> bool:
    """Whether a message is a compaction summary (for UI rendering)."""
    return (
        msg.role == "user"
        and msg.content is not None
        and msg.content.startswith(f"<{SUMMARY_TAG}>")
    )
