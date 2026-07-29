"""Tests for the per-file mutation queue (limbo.tools.mutation_queue)."""

from __future__ import annotations

import threading
from pathlib import Path

from limbo.tools.mutation_queue import mutation_lock_for


def test_same_path_returns_same_lock(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    assert mutation_lock_for(target) is mutation_lock_for(target)


def test_different_paths_return_different_locks(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("x")
    assert mutation_lock_for(a) is not mutation_lock_for(b)


def test_missing_file_resolves_lexically(tmp_path: Path):
    missing = tmp_path / "new.txt"
    assert mutation_lock_for(missing) is mutation_lock_for(missing)


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
