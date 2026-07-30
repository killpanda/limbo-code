"""Tests for the steer queue (mid-turn user messages, RFC LIM-20).

The queue owns queueing semantics — id generation, FIFO order, cancel-by-id
up to the drain boundary; the Agent owns only *when* to drain.
"""

from __future__ import annotations

from limbo.models import Attachment
from limbo.steer import SteerQueue


def test_offer_returns_item_with_unique_ids():
    queue = SteerQueue()
    first = queue.offer("hello")
    second = queue.offer("world")
    assert first.text == "hello"
    assert first.id and second.id
    assert first.id != second.id


def test_offer_defaults_attachments_to_empty():
    queue = SteerQueue()
    item = queue.offer("hi")
    assert item.attachments == []
    attached = queue.offer("pic", [Attachment(path="/tmp/a.png", name="a", kind="image")])
    assert len(attached.attachments) == 1


def test_drain_returns_fifo_and_clears():
    queue = SteerQueue()
    queue.offer("one")
    queue.offer("two")
    drained = queue.drain()
    assert [item.text for item in drained] == ["one", "two"]
    assert len(queue) == 0
    assert queue.drain() == []


def test_cancel_removes_queued_item():
    queue = SteerQueue()
    keep = queue.offer("keep")
    drop = queue.offer("drop")
    cancelled = queue.cancel(drop.id)
    assert cancelled is not None and cancelled.text == "drop"
    assert [item.id for item in queue.drain()] == [keep.id]


def test_cancel_unknown_or_drained_id_returns_none():
    queue = SteerQueue()
    item = queue.offer("gone")
    queue.drain()
    # Past the drain boundary the item is part of history and cannot be
    # retracted; an unknown id behaves the same.
    assert queue.cancel(item.id) is None
    assert queue.cancel("no-such-id") is None


def test_cancel_latest_pops_newest_first():
    queue = SteerQueue()
    queue.offer("older")
    newer = queue.offer("newer")
    assert queue.cancel_latest() == newer
    assert len(queue) == 1
    assert queue.cancel_latest() is not None
    assert queue.cancel_latest() is None
