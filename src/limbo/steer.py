"""Steer queue: user messages submitted while the agent is mid-turn (RFC LIM-20).

The queue owns queueing semantics — id generation, FIFO order, and the
cancel boundary (an item can be cancelled by id only until it is drained;
past that point it is part of the conversation history and cannot be
retracted). The Agent owns the other half: *when* to drain (loop top,
turn end, run head — the history-consistency points), the tracing, and
journaling the queue into the session file so it survives a crash.

Two modes (pi-durable's inbox): a ``steer`` message is injected at the
next consistency point of the running turn; a ``follow_up`` waits until
the current task (including a goal loop) is done and then starts a turn.
"""

from __future__ import annotations

import collections
import secrets
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal

from limbo.models import Attachment

SteerMode = Literal["steer", "follow_up"]
STEER: SteerMode = "steer"
FOLLOW_UP: SteerMode = "follow_up"


@dataclass(frozen=True)
class SteerItem:
    """A user message queued while the agent is mid-turn."""

    id: str
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    mode: SteerMode = STEER
    # "user", or "subagent" for a notice delivered to the main agent; refs
    # are the notice ids it carries (dedupe across a crash).
    source: str = "user"
    refs: tuple[str, ...] = ()


class SteerQueue:
    """FIFO queue of mid-turn user messages.

    A deque (not asyncio.Queue) so individual items can be cancelled by id;
    producers/consumers only ever use non-blocking ops on the single
    Textual event loop.
    """

    def __init__(self) -> None:
        self._items: collections.deque[SteerItem] = collections.deque()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> collections.abc.Iterator[SteerItem]:
        """Read-only view of the queued items (FIFO); does not drain."""
        return iter(tuple(self._items))

    def offer(
        self,
        text: str,
        attachments: list[Attachment] | None = None,
        mode: SteerMode = STEER,
        *,
        source: str = "user",
        refs: tuple[str, ...] = (),
    ) -> SteerItem:
        """Queue a message; returns the item (the UI binds its card to the id)."""
        item = SteerItem(
            id=secrets.token_hex(4),
            text=text,
            attachments=attachments or [],
            mode=mode,
            source=source,
            refs=tuple(refs),
        )
        self._items.append(item)
        return item

    def restore(self, items: list[SteerItem]) -> None:
        """Put items recovered from the session file back, in order."""
        self._items.extend(items)

    def cancel(self, item_id: str) -> SteerItem | None:
        """Remove and return a still-queued item.

        None when the id is unknown or the item was already drained (past
        that boundary it is part of history and cannot be retracted).
        """
        for item in self._items:
            if item.id == item_id:
                self._items.remove(item)
                return item
        return None

    def cancel_latest(self) -> SteerItem | None:
        """Remove and return the newest queued item, if any."""
        if not self._items:
            return None
        return self._items.pop()

    def drain(self, modes: Collection[SteerMode] = (STEER, FOLLOW_UP)) -> list[SteerItem]:
        """Remove and return the queued items of ``modes`` in FIFO order."""
        items = [item for item in self._items if item.mode in modes]
        for item in items:
            self._items.remove(item)
        return items
