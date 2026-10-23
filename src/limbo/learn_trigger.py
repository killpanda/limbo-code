"""When daily mode starts an automatic skill review (/learn) by itself.

The skill-reviewer is a fork subagent: every run re-reads the whole
conversation, so running it after every busy turn is expensive and mostly
ends in "Nothing to save". The trigger watches the main conversation's
events and only fires when all of these hold:

- the turn finished cleanly (no interrupt, no error);
- enough tool *rounds* (LLM responses that called tools; a parallel batch
  is one round, as in hermes-agent's ``_iters_since_skill``) ran since the
  last skill update or review;
- at least ``cooldown`` turns passed since the last review;
- a signal suggests there is something to learn: the user corrected the
  assistant, a tool failed and the retry succeeded, or a skill was used.
  Without a signal, three times the threshold still fires as a fallback.

A review that ends in "Nothing to save" doubles the threshold (capped at
``MAX_BACKOFF``×); one that changed a skill brings it back to the base.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from limbo.agent import CompactionEvent, SteerEvent, ToolCallRequest, ToolResultEvent
from limbo.models import Message

if TYPE_CHECKING:
    from limbo.subagents.board import Task

MAX_BACKOFF = 8
# Rounds (× threshold) after which a review runs without any signal.
FALLBACK_FACTOR = 3
# How the reviewer says it changed nothing (its brief asks for the first).
NOTHING_SAVED = ("nothing to save", "没有需要保存", "无需保存")

# The user telling the assistant how things should be done. Narrow on
# purpose: a missed correction only delays a review until the fallback.
_CORRECTION = re.compile(
    r"不要|别再|别用|不用再|太长|太短|太啰嗦|啰嗦|以后|下次|每次都|总是|记住|"
    r"不对|错了|不是这样|"
    r"\b(?:don'?t|do not|stop doing|always|never|next time|from now on|wrong)\b|"
    r"\btoo (?:long|short|verbose|wordy)\b",
    re.IGNORECASE,
)
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search(_REMINDER.sub("", text)))


def user_texts(messages: Sequence[Message]) -> list[str]:
    """What the user typed, minus injected reminders and subagent updates."""
    texts = []
    for m in messages:
        content = (m.content or "").strip()
        if m.role == "user" and content and not content.startswith("<subagent_updates"):
            texts.append(content)
    return texts


class LearnTrigger:
    """Per-session state for the automatic skill review (see module doc)."""

    def __init__(self, every: int, cooldown: int) -> None:
        self.every = every
        self.cooldown = cooldown
        self.rounds = 0
        # Start past the cooldown: the first review needs no wait.
        self.turns_since_review = cooldown
        self.backoff = 1
        self.signals: set[str] = set()
        self._review_id: str | None = None
        # Per turn: inside a run of tool calls (one round), tools that
        # failed this turn, and whether the turn did anything at all.
        self._in_round = False
        self._failed: set[str] = set()
        self._active = False

    @property
    def threshold(self) -> int:
        return self.every * self.backoff

    def observe(self, event: object) -> None:
        """Feed one event of the main conversation's turn."""
        if isinstance(event, ToolCallRequest):
            self._active = True
            if not self._in_round:
                self.rounds += 1
            self._in_round = True
            return
        self._in_round = False
        if not isinstance(event, CompactionEvent):
            self._active = True
        if isinstance(event, ToolResultEvent):
            self._observe_result(event)
        elif isinstance(event, SteerEvent):
            self.observe_user_text(event.text)

    def _observe_result(self, event: ToolResultEvent) -> None:
        if not event.result.success:
            self._failed.add(event.name)
            return
        if event.name == "skill_manage":
            # The main agent updated a skill itself: nothing new to review.
            self.rounds = 0
            self.signals.clear()
        elif event.name in self._failed:
            self.signals.add("recovered")
        if event.name == "read" and str(event.arguments.get("path", "")).endswith("SKILL.md"):
            self.signals.add("skill_used")

    def observe_user_text(self, text: str) -> None:
        if is_correction(text):
            self.signals.add("correction")

    def end_turn(self, state: str) -> bool:
        """The turn ended in ``state``; True when a review should start."""
        active, self._active = self._active, False
        self._in_round = False
        self._failed.clear()
        if not active:
            return False  # /compact and other runs without a model turn
        self.turns_since_review += 1
        if state != "done" or self.every <= 0:
            return False
        if self.turns_since_review < self.cooldown or self.rounds < self.threshold:
            return False
        return bool(self.signals) or self.rounds >= self.threshold * FALLBACK_FACTOR

    def review_started(self, task_id: str | None) -> None:
        """A review (automatic or /learn) started: count afresh from here."""
        self.rounds = 0
        self.turns_since_review = 0
        self.signals.clear()
        self._review_id = task_id

    def settle(self, tasks: dict[str, Task]) -> None:
        """Adjust the backoff once the last review has an outcome."""
        task = tasks.get(self._review_id or "")
        if task is None or task.outcome is None:
            return
        self._review_id = None
        if task.outcome.status != "done":
            return  # failed or cancelled: says nothing about the session
        summary = task.outcome.summary.lower()
        if any(phrase in summary for phrase in NOTHING_SAVED):
            self.backoff = min(self.backoff * 2, MAX_BACKOFF)
        else:
            self.backoff = 1
