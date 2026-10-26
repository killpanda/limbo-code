"""Bounded garbage-collector pauses for long sessions.

CPython's full (gen2) collection walks every tracked object. A long
conversation keeps hundreds of thousands of them alive (widgets, rendered
strips, history), so each full collection became a whole-UI freeze that
grew with the session: measured 140 ms after one turn, 500-750 ms after ten,
several times per turn.

``HeapKeeper`` ticks every few seconds: it collects what was allocated since
the last tick, then freezes the survivors (``gc.freeze``) so later
collections skip them. A pause is then bounded by one tick's allocations
instead of the session's size.

Frozen objects are still freed by reference counting; only reference
cycles among them (widgets removed after they were frozen) wait. Once the
app has been quiet for a while — almost nothing allocated over several
ticks — one full sweep (unfreeze + collect + freeze) reclaims them; nobody
is interacting then, so that one long pause goes unnoticed. ``release``
hands everything back right away when a whole conversation is dropped.
"""

from __future__ import annotations

import gc

# Young-object settle cadence.
TICK_SECONDS = 3.0
# Ticks in a row with fewer new objects than this count as quiet.
QUIET_OBJECTS = 5_000
QUIET_TICKS = 6


class HeapKeeper:
    def __init__(self) -> None:
        self._quiet_ticks = 0
        # Busy-time objects were frozen since the last full sweep.
        self._dirty = False

    def tick(self) -> None:
        # Frozen objects are not listed: this is what is new since the
        # last freeze (cheap — the young part of the heap only).
        if len(gc.get_objects()) >= QUIET_OBJECTS:
            self._quiet_ticks = 0
            self._dirty = True
        else:
            self._quiet_ticks += 1
        if self._dirty and self._quiet_ticks >= QUIET_TICKS:
            # Quiet: one full sweep per quiet period.
            gc.unfreeze()
            self._dirty = False
        gc.collect()
        gc.freeze()


def release() -> None:
    """Hand frozen objects back to the collector (a conversation was dropped)."""
    gc.unfreeze()
