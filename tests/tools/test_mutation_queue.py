"""Tests for the per-file mutation queue (limbo.tools.mutation_queue)."""

from __future__ import annotations

import threading
from pathlib import Path

from limbo.tools.mutation_queue import mutation_lock_for


def test_lock_actually_serializes_threads(tmp_path: Path):
    lock = mutation_lock_for(tmp_path / "f.txt")
    order: list[str] = []

    def worker(name: str, delay: float) -> None:
        import time

        with lock:
            time.sleep(delay)
            order.append(name)

    t1 = threading.Thread(target=worker, args=("first", 0.05))
    t2 = threading.Thread(target=worker, args=("second", 0.0))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert order == ["first", "second"]
