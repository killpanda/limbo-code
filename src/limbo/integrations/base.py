"""Shared contract for terminal multiplexer / agent orchestrator integrations.

Every integration (Herdr, OSC 7501, future *mux tools, ...) implements
``IntegrationReporter``: it receives semantic lifecycle state
(``working``/``idle``/``blocked``/``done``/``error``) plus session
identity, and maps that onto whatever the specific tool understands.
Call sites (UI, agent loop) only ever talk to ``CompositeReporter`` —
adding a new integration means writing one module here and registering it
in ``create_reporters()``, with zero changes at call sites.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol

from limbo.agent import ErrorEvent, InterruptEvent

# Semantic lifecycle states every integration must be able to represent.
# ``blocked`` means the agent is waiting on a user decision; ``done`` and
# ``error`` are how a turn ended (a tool without them maps both to idle).
AgentState = Literal["working", "idle", "blocked", "done", "error"]


class TurnOutcome:
    """Folds one pump run's events into the state reported when it ends.

    A clean finish is ``done``, an ``ErrorEvent`` (always turn-ending) or a
    crash is ``error``, and a user interrupt is plain ``idle`` — the user
    already knows, so there is no unseen result to flag.
    """

    def __init__(self) -> None:  # noqa: D107
        self.state: AgentState = "done"
        self.message: str | None = None

    def observe(self, event: object) -> None:
        if isinstance(event, InterruptEvent):
            self.state, self.message = "idle", None
        elif isinstance(event, ErrorEvent) and self.state != "idle":
            self.fail(event.message)

    def fail(self, message: str) -> None:
        self.state, self.message = "error", message


class IntegrationReporter(Protocol):
    """One external tool interested in Limbo's lifecycle.

    Implementations must be fire-and-forget: never block, never raise.
    """

    def report(
        self,
        state: AgentState,
        *,
        message: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
    ) -> None:
        """Report lifecycle state (optionally carrying session identity)."""
        ...

    def report_session(
        self, session_id: str, session_path: str | None = None
    ) -> None:
        """Report session identity when it changes independently of state."""
        ...

    def release(self) -> None:
        """Release any held authority/resources (on agent exit)."""
        ...


class CompositeReporter:
    """Fans lifecycle reports out to every active integration.

    Always safe to call, even with zero reporters — call sites never need
    to know which (if any) integrations are active.
    """

    def __init__(self, reporters: Iterable[IntegrationReporter] = ()):  # noqa: D107
        self._reporters = list(reporters)

    def __bool__(self) -> bool:
        return bool(self._reporters)

    def report(
        self,
        state: AgentState,
        *,
        message: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
    ) -> None:
        for reporter in self._reporters:
            reporter.report(
                state,
                message=message,
                session_id=session_id,
                session_path=session_path,
            )

    def report_session(
        self, session_id: str, session_path: str | None = None
    ) -> None:
        for reporter in self._reporters:
            reporter.report_session(session_id, session_path)

    def release(self) -> None:
        for reporter in self._reporters:
            reporter.release()
