"""Tests for the steer queue (mid-turn user messages, RFC LIM-20).

The queue owns queueing semantics — id generation, FIFO order, cancel-by-id
up to the drain boundary; the Agent owns only *when* to drain.
"""

from __future__ import annotations

from limbo.steer import SteerQueue


def test_drain_by_mode_leaves_follow_ups_queued():
    queue = SteerQueue()
    queue.offer("steer one")
    later = queue.offer("after the task", mode="follow_up")
    queue.offer("steer two")

    drained = queue.drain(("steer",))

    assert [i.text for i in drained] == ["steer one", "steer two"]
    assert list(queue) == [later]
    assert queue.drain() == [later]


