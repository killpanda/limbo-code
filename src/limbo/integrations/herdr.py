"""Herdr terminal multiplexer integration.

When Limbo runs inside a Herdr pane, it inherits ``HERDR_ENV``,
``HERDR_PANE_ID``, ``HERDR_BIN_PATH``, and ``HERDR_SOCKET_PATH``. This
reporter uses Herdr's CLI to report semantic lifecycle state
(``working``/``idle``/``blocked``) and native session identity, per
https://herdr.dev/docs/integrations/#integrate-your-own-agent

The integration is a strict no-op outside Herdr: ``from_env()`` returns
``None`` unless ``HERDR_ENV=1`` and both the pane id and a usable ``herdr``
binary (``HERDR_BIN_PATH``, else PATH) are present.
Reports are fire-and-forget CLI invocations — they must never block or
break the UI, and a missing/broken ``herdr`` binary is silently ignored.

``--source`` must stay stable and unique to this integration; ``--seq``
must be strictly increasing — Herdr keeps a per-(pane, source) watermark
and silently drops anything at or below it, including ``release-agent``.
The watermark survives process restarts, so the sequence starts from the
wall clock (nanoseconds), not 1: a new Limbo process in a pane that
previously ran one would otherwise never clear the old watermark.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping

from limbo.integrations.base import AgentState

SOURCE = "custom:limbo"
AGENT = "limbo"


class HerdrReporter:
    """Fire-and-forget lifecycle reporter bound to one Herdr pane."""

    def __init__(self, pane_id: str, bin_path: str, start_seq: int | None = None):
        self._pane_id = pane_id
        self._bin_path = bin_path
        # Start from the wall clock so a later process in the same pane
        # clears the previous one's watermark (see module docstring).
        self._seq = start_seq if start_seq is not None else time.time_ns()
        self._released = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HerdrReporter | None:
        """Create a reporter when running inside a Herdr pane, else None.

        ``HERDR_BIN_PATH`` is preferred (portable across PATH differences)
        but is not guaranteed to be set — fall back to a PATH lookup.
        """
        env = os.environ if env is None else env
        if env.get("HERDR_ENV") != "1":
            return None
        pane_id = env.get("HERDR_PANE_ID")
        bin_path = env.get("HERDR_BIN_PATH") or shutil.which("herdr")
        if not pane_id or not bin_path:
            return None
        return cls(pane_id, bin_path)

    def report(
        self,
        state: AgentState,
        *,
        message: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
    ) -> None:
        """Report lifecycle state (optionally with session identity)."""
        cmd = [
            self._bin_path,
            "pane",
            "report-agent",
            self._pane_id,
            "--source",
            SOURCE,
            "--agent",
            AGENT,
            "--state",
            state,
            "--seq",
            str(self._next_seq()),
        ]
        if message:
            cmd += ["--message", message]
        if session_id:
            cmd += ["--agent-session-id", session_id]
        if session_path:
            cmd += ["--agent-session-path", session_path]
        self._run(cmd)

    def report_session(
        self, session_id: str, session_path: str | None = None
    ) -> None:
        """Report session identity when it changes independently of state
        (``/new``, ``/sessions`` switch)."""
        cmd = [
            self._bin_path,
            "pane",
            "report-agent-session",
            self._pane_id,
            "--source",
            SOURCE,
            "--agent",
            AGENT,
            "--seq",
            str(self._next_seq()),
            "--agent-session-id",
            session_id,
        ]
        if session_path:
            cmd += ["--agent-session-path", session_path]
        self._run(cmd)

    def release(self) -> None:
        """Release this source's lifecycle authority (on agent exit).

        Idempotent: ``on_unmount``, ``atexit``, and signal handlers all
        call this on the way out; only the first call reports.
        """
        if self._released:
            return
        self._released = True
        # seq matters here: Herdr drops a release at or below this source's
        # watermark, so a seq-less release after sequenced reports is
        # silently ignored.
        self._run(
            [
                self._bin_path,
                "pane",
                "release-agent",
                self._pane_id,
                "--source",
                SOURCE,
                "--agent",
                AGENT,
                "--seq",
                str(self._next_seq()),
            ]
        )

    def _next_seq(self) -> int:
        # Assigned synchronously before spawning, so sequence order matches
        # call order even when deliveries race.
        self._seq += 1
        return self._seq

    @staticmethod
    def _run(cmd: list[str]) -> None:
        try:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            # A missing/broken herdr binary must never break the session.
            pass
