"""Integrations with external terminal multiplexers / agent orchestrators.

Each integration lives in its own module (e.g. ``herdr``) and implements
the ``IntegrationReporter`` protocol from ``base``. Its ``from_env()``
returns ``None`` when Limbo is not running inside that tool, so the
integration is a strict no-op elsewhere. ``create_reporters()`` collects
every active integration into one ``CompositeReporter`` — the only object
call sites ever touch.
"""

from __future__ import annotations

import atexit
import os
import signal
from collections.abc import Mapping
from types import FrameType

from limbo.integrations.base import (
    AgentState,
    CompositeReporter,
    IntegrationReporter,
)
from limbo.integrations.herdr import HerdrReporter

__all__ = [
    "AgentState",
    "CompositeReporter",
    "HerdrReporter",
    "IntegrationReporter",
    "create_reporters",
    "install_exit_hooks",
]


def create_reporters(env: Mapping[str, str] | None = None) -> CompositeReporter:
    """Detect active integrations from the environment.

    Register new integrations here: append ``XReporter.from_env(env)`` when
    it returns a reporter. Order is irrelevant; each reporter is
    independent and fire-and-forget.
    """
    env = os.environ if env is None else env
    reporters: list[IntegrationReporter] = []
    if (herdr := HerdrReporter.from_env(env)) is not None:
        reporters.append(herdr)
    return CompositeReporter(reporters)


def install_exit_hooks(reporters: CompositeReporter) -> None:
    """Release integration authority on every process-exit path.

    A clean Textual quit releases via ``MainScreen.on_unmount``, but that
    never runs when the process dies by signal — e.g. the pane is closed
    or killed (SIGHUP/SIGTERM) — and Herdr would keep showing the agent
    forever. Covered here: normal interpreter exit (``atexit``, which also
    catches ``SystemExit`` and unhandled exceptions) and SIGHUP/SIGTERM
    (released, then re-raised with the default disposition so the process
    still dies by the original signal). Reporters are idempotent, so
    several hooks firing on one exit path is harmless.
    """
    if not reporters:
        return
    atexit.register(reporters.release)

    def _release_and_reraise(signum: int, frame: FrameType | None) -> None:
        reporters.release()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for name in ("SIGHUP", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:  # SIGHUP does not exist on Windows
            signal.signal(signum, _release_and_reraise)
