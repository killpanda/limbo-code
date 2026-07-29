"""Per-file serialization for concurrent mutations.

With parallel tool execution, write/edit calls targeting the same file
must not interleave. Tools run synchronously inside ``asyncio.to_thread``
worker threads, so an ``asyncio.Lock`` cannot be acquired at the point of
use; a module-level ``threading.Lock`` keyed by canonical path is the
correct layer (same shape as pi's file-mutation-queue, which also keys by
realpath). Different files still proceed in parallel.
"""

from __future__ import annotations

import threading
from pathlib import Path

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _canonical_key(path: Path) -> str:
    try:
        return str(path.resolve(strict=True))
    except OSError:
        # File does not exist yet (write to a new path): resolve lexically.
        return str(path.resolve(strict=False))


def mutation_lock_for(path: Path) -> threading.Lock:
    """Return the shared lock serializing mutations to ``path``."""
    key = _canonical_key(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock
