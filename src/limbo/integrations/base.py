"""Shared contract for terminal multiplexer / agent orchestrator integrations.

Every integration (Herdr, future *mux tools, ...) implements
``IntegrationReporter``: it receives semantic lifecycle state
(``working``/``idle``/``blocked``) plus session identity, and maps that onto
whatever the specific tool understands. Call sites (UI, agent loop) only
ever talk to ``CompositeReporter`` — adding a new integration means writing
one module here and registering it in ``create_reporters()``, with zero
changes at call sites.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol

# Semantic lifecycle states every integration must be able to represent.
# ``blocked`` means the agent is waiting on a user decision.
AgentState = Literal["working", "idle", "blocked"]


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
