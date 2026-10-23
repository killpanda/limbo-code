"""Program status via OSC 7501 (terminal-native lifecycle reporting).

Spec: https://www.superlogical.com/rex/docs/build/program-status (rev 0.4).
Limbo reports one *root* record (no ``id``) through the pty, so terminals
that implement the protocol can show working/blocked/done badges, notify
when a turn finishes, etc. — over SSH and in containers, with no sockets.

A report fully replaces the record, so ``app`` is resent every time.
``msg`` is base64 UTF-8, one line, control characters stripped (the
terminal discards a report whose decoded text contains any).

Guarding: unknown OSCs are ignored by well-behaved terminals, but Limbo
still only emits when the frontend hands over a terminal writer (TTY
output, not ``--json``), ``TERM`` is not ``dumb``, and the user has not
opted out with ``LIMBO_PROGRAM_STATUS=0``. Inside tmux the sequence is
wrapped in DCS passthrough (needs ``set -g allow-passthrough on``;
otherwise tmux drops it silently). No feature query is sent: detection is
optional per spec, and reading the reply would race the frontend's input
reader.
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Callable, Mapping

from limbo.integrations.base import AgentState

APP = "limbo"
# Spec limit: 2048 decoded bytes; stay well under it.
_MSG_MAX_CHARS = 200
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_OFF = {"0", "off", "false", "no"}

TerminalWriter = Callable[[str], None]


def encode_report(pairs: Mapping[str, str]) -> str:
    """``ESC ] 7501 ; k=v:k=v ESC \\`` for already-valid values."""
    body = ":".join(f"{key}={value}" for key, value in pairs.items())
    return f"\x1b]7501;{body}\x1b\\"


def tmux_passthrough(seq: str) -> str:
    """Wrap ``seq`` so tmux forwards it to the outer terminal."""
    return "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"


def _b64_line(text: str) -> str | None:
    line = " ".join(_CONTROL.sub(" ", text).split())[:_MSG_MAX_CHARS]
    return base64.b64encode(line.encode()).decode() if line else None


class OscStatusReporter:
    """Writes OSC 7501 program-status reports to the controlling terminal."""

    def __init__(self, write: TerminalWriter, *, tmux: bool = False):
        self._write = write
        self._tmux = tmux
        self._released = False

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None, write: TerminalWriter | None
    ) -> OscStatusReporter | None:
        env = os.environ if env is None else env
        if write is None:
            return None
        if env.get("LIMBO_PROGRAM_STATUS", "").strip().lower() in _OFF:
            return None
        if env.get("TERM") == "dumb":
            return None
        return cls(write, tmux=bool(env.get("TMUX")))

    def report(
        self,
        state: AgentState,
        *,
        message: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
    ) -> None:
        pairs = {"state": state, "app": APP}
        if state == "blocked":
            # Limbo's only blocked state is answering a question (the
            # /goal verify-command confirmation).
            pairs["kind"] = "question"
        if message and (msg := _b64_line(message)):
            pairs["msg"] = msg
        self._send(pairs)

    def report_session(
        self, session_id: str, session_path: str | None = None
    ) -> None:
        """Session identity has no OSC 7501 key; nothing to report."""

    def release(self) -> None:
        """Clear our record on exit (idempotent across exit hooks)."""
        if self._released:
            return
        self._released = True
        self._send({"state": "clear"})

    def _send(self, pairs: Mapping[str, str]) -> None:
        seq = encode_report(pairs)
        try:
            self._write(tmux_passthrough(seq) if self._tmux else seq)
        except Exception:  # noqa: BLE001 - reporters must never raise
            pass
