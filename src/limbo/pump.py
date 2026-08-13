"""Turn pump: the non-UI orchestrator that runs turns back-to-back.

Sits between a frontend (the TUI, or a future headless/CLI frontend) and
``Agent``. ``Agent.run()`` is a lazy async generator — nothing happens
until someone pulls events out of it. This module is that someone: it
owns

- **single-flight** — one ``_running`` flag covers turns, goal rounds,
  follow-ups, and ``/compact`` mini-turns; set eagerly at call time so
  there is no race window before the first event;
- **the steer follow-up drain** (RFC LIM-20/53) — steers left queued at a
  turn boundary become the next turn's input, joined. The supervisor
  drain (``_supervise``) is the only follow-up site: frontends never
  drain. An interrupted turn never auto-follows-up — leftovers stay
  queued for the next ``run()``'s head fallback;
- **the goal closed loop** (RFC LIM-40 v1.2) — at a quiet turn boundary,
  run the verify command and feed failures back verbatim into the next
  round, while ``agent.py`` stays untouched: every round is a complete
  ``Agent.run()`` turn, so per-round ``max_iterations``, loop-top
  auto-compaction, steer injection points, and turn tracing all behave
  exactly as before.

Non-UI layer: this module must never import textual (locked by a test).
Frontends are adapters translating its events into UI updates.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from limbo.agent import Agent, AgentEvent, SteerEvent
from limbo.goal import (
    STATUS_ACTIVE,
    STATUS_CLEARED,
    STATUS_EXHAUSTED,
    GoalState,
    VerifyResult,
    build_goal_prompt,
    build_wrapup_prompt,
    next_state,
    run_verify,
)
from limbo.models import Attachment

# -- goal events (frontend renders these; headless prints them) ---------------


@dataclass(frozen=True)
class GoalVerifyStarted:
    command: str
    round: int
    max_rounds: int


@dataclass(frozen=True)
class GoalVerifyResultEvent:
    result: VerifyResult
    round: int
    max_rounds: int


@dataclass(frozen=True)
class GoalPassed:
    rounds: int


@dataclass(frozen=True)
class GoalExhausted:
    rounds: int


@dataclass(frozen=True)
class GoalResumed:
    """An exhausted goal was reactivated by a user message (budget reset)."""

    max_rounds: int


GoalEvent = (
    GoalVerifyStarted
    | GoalVerifyResultEvent
    | GoalPassed
    | GoalExhausted
    | GoalResumed
)

PumpEvent = AgentEvent | GoalEvent


class TurnPump:
    """Pumps one session's agent: turns, follow-ups, compact, goal loop.

    Single-flight: ``run()``/``compact()`` raise if a pump is already in
    progress; the ``running`` flag is set eagerly (plain methods returning
    the async iterator — an async-generator body would not execute until
    the first ``__anext__``, leaving a race window). The flag is released
    in the supervisor's finally, in the same await-free section as the
    last drain check, so a submission interleaving with pump teardown is
    always seen by exactly one side (queued-steer vs new turn).
    """

    def __init__(
        self,
        agent: Agent,
        workdir: Path,
        *,
        max_rounds: int = 10,
        verify_timeout_ms: int = 600_000,
    ) -> None:
        self._agent = agent
        self._workdir = workdir
        self._max_rounds = max_rounds
        self._verify_timeout_ms = verify_timeout_ms
        self._running = False
        self._verify_task: asyncio.Task[VerifyResult] | None = None
        # Adopt the configured budget onto a restored goal (D4/D6: silent
        # resume keeps the persisted state, only the budget is refreshed).
        state = agent.session_meta.goal
        if state is not None and state.max_rounds != max_rounds:
            agent.session_meta.goal = state.model_copy(
                update={"max_rounds": max_rounds}
            )

    # -- state surface (used by the /goal command variants) -------------------

    @property
    def running(self) -> bool:
        """Busy: a turn, goal loop, follow-up, or compact owns the agent."""
        return self._running

    @property
    def verifying(self) -> bool:
        return self._verify_task is not None and not self._verify_task.done()

    def status(self) -> GoalState | None:
        return self._agent.session_meta.goal

    def _save_state(self, state: GoalState) -> None:
        # Persisted via the session meta on the agent's next per-turn save.
        self._agent.session_meta.goal = state

    def set_goal(self, text: str, verify_command: str = "") -> GoalState:
        state = GoalState(
            text=text,
            verify_command=verify_command,
            max_rounds=self._max_rounds,
        )
        self._save_state(state)
        self._agent.trace.log(
            "goal_set",
            text=text,
            verify_command=verify_command,
            max_rounds=self._max_rounds,
        )
        return state

    def set_verify(self, command: str) -> GoalState | None:
        """Set/replace the verify command; a fresh gate resets the budget.

        Called by the frontend after the user confirms a model proposal
        (or, in a headless frontend, programmatically)."""
        state = self.status()
        if state is None or state.status == STATUS_CLEARED:
            return None
        state = state.model_copy(
            update={
                "verify_command": command,
                "rounds_completed": 0,
                "status": STATUS_ACTIVE,
            }
        )
        self._save_state(state)
        return state

    def clear(self) -> None:
        """Exit goal mode. Never interrupts the in-flight turn; an in-flight
        verify is cancelled so the loop stops at the next boundary."""
        state = self.status()
        if state is None or state.status == STATUS_CLEARED:
            return
        self._save_state(state.model_copy(update={"status": STATUS_CLEARED}))
        self._agent.trace.log("goal_cleared")
        self.cancel_verify()

    def cancel_verify(self) -> bool:
        """Cancel an in-flight verify subprocess (Esc route); no-op otherwise."""
        if self.verifying and self._verify_task is not None:
            self._verify_task.cancel()
            return True
        return False

    # -- the pumps --------------------------------------------------------------

    def run(
        self, user_input: str, attachments: list[Attachment] | None = None
    ) -> AsyncIterator[PumpEvent]:
        if self._running:
            raise RuntimeError("TurnPump.run called while already running")
        # Eager (see class docstring): the flag must be set before the
        # caller awaits the first event, or single-flight has a gap.
        self._running = True
        return self._supervise(user_input, attachments)

    def compact(self) -> AsyncIterator[AgentEvent]:
        """Pump one manual compaction under the same single-flight flag.

        Auto-compaction fires inside ``Agent.run()`` at loop top; this is
        the ``/compact`` path. Sharing ``_running`` with ``run()`` makes
        "the session is busy" one flag instead of two.
        """
        if self._running:
            raise RuntimeError("TurnPump.compact called while already running")
        self._running = True
        return self._compact_pump()

    async def _compact_pump(self) -> AsyncIterator[AgentEvent]:
        try:
            async for event in self._agent.compact(trigger="manual"):
                yield event
        finally:
            self._running = False

    async def _supervise(
        self, user_input: str, attachments: list[Attachment] | None
    ) -> AsyncIterator[PumpEvent]:
        """Run goal rounds; turn leftovers into follow-up turns.

        The follow-up drain lives here and nowhere else (RFC LIM-20):
        ``_run_once`` drains between rounds (user steers outrank goal
        continuation); this supervisor catches whatever is queued past its
        last boundary — steers that arrived during verify / the wrap-up
        turn / the teardown window — and pumps one more turn for them.
        """
        next_input = user_input
        next_attachments = attachments
        try:
            while True:
                async for event in self._run_once(next_input, next_attachments):
                    yield event
                next_attachments = None

                # RFC LIM-53: an interrupted pump never auto-follows-up —
                # leftovers stay queued for the next run()'s head fallback.
                if self._agent.was_interrupted:
                    return

                leftovers = self._agent.drain_steer()
                if not leftovers:
                    return
                # E1: one SteerEvent per item so the frontend flips the
                # queued cards by id. Attachments ride along (the old
                # frontend-side drain preserved them too).
                for item in leftovers:
                    yield SteerEvent(id=item.id, text=item.text)
                next_input = "\n\n".join(item.text for item in leftovers)
                next_attachments = [
                    a for item in leftovers for a in item.attachments
                ]
        finally:
            self._running = False

    async def _run_once(
        self, user_input: str, attachments: list[Attachment] | None
    ) -> AsyncIterator[PumpEvent]:
        """One pass of the goal closed loop: turn → verify → next round.

        Returns when no goal continuation applies (no active goal, goal
        passed/exhausted/cleared, or user interrupt); the supervisor owns
        what happens next.
        """
        # D3/LIM-35: any user message reactivates an exhausted goal with
        # a fresh budget. (/goal clear is the only way out for good.)
        state = self.status()
        if state is not None and state.status == STATUS_EXHAUSTED:
            state = state.model_copy(
                update={"status": STATUS_ACTIVE, "rounds_completed": 0}
            )
            self._save_state(state)
            self._agent.trace.log("goal_resumed", max_rounds=state.max_rounds)
            yield GoalResumed(max_rounds=state.max_rounds)

        next_input = user_input
        next_attachments = attachments
        while True:
            async for event in self._agent.run(next_input, next_attachments):
                yield event
            next_attachments = None

            # User interrupt (RFC LIM-53) outranks everything at the
            # turn boundary: stop the loop BEFORE draining leftover
            # steers — queued messages stay queued and the goal stays
            # active (same posture as cancel_verify: the user called
            # stop, not continue).
            # Known boundary: ESC landing in the narrow window where
            # running=True but the next run() has not started yet sets
            # a flag the next run() resets at entry — silently a no-op.
            # Accepted (nothing is in flight to interrupt).
            if self._agent.was_interrupted:
                self._agent.trace.log("goal_interrupted")
                return

            # Turn boundary, user first: leftover steer messages (error
            # exits) outrank goal continuation. E1: yield one SteerEvent
            # per item so the frontend flips the queued cards by id.
            leftovers = self._agent.drain_steer()
            if leftovers:
                for item in leftovers:
                    yield SteerEvent(id=item.id, text=item.text)
                next_input = "\n\n".join(item.text for item in leftovers)
                continue

            state = self.status()
            if (
                state is None
                or state.status != STATUS_ACTIVE
                or not state.verify_command
            ):
                return

            round_no = state.rounds_completed + 1
            yield GoalVerifyStarted(
                command=state.verify_command,
                round=round_no,
                max_rounds=state.max_rounds,
            )
            result = await self._run_verify(state.verify_command)
            self._agent.trace.log(
                "goal_verify",
                command=state.verify_command,
                exit_code=result.exit_code,
                round=round_no,
                timed_out=result.timed_out,
                cancelled=result.cancelled,
            )
            yield GoalVerifyResultEvent(
                result=result, round=round_no, max_rounds=state.max_rounds
            )
            if result.cancelled:
                # User interrupt (Esc/clear): stop the loop, goal stays
                # active.
                return

            state = next_state(state, result)
            self._save_state(state)
            if state.status == "passed":
                self._agent.trace.log(
                    "goal_complete", rounds=state.rounds_completed
                )
                yield GoalPassed(rounds=state.rounds_completed)
                return
            if state.status == STATUS_EXHAUSTED:
                self._agent.trace.log(
                    "goal_exhausted", rounds=state.rounds_completed
                )
                # LIM-37 wrap-up: a normal (but uncounted, unverified)
                # turn producing the summary + recovery guidance.
                async for event in self._agent.run(build_wrapup_prompt(state)):
                    yield event
                yield GoalExhausted(rounds=state.rounds_completed)
                return

            self._agent.trace.log(
                "goal_round", next_round=state.rounds_completed + 1
            )
            next_input = build_goal_prompt(state, result)

    async def _run_verify(self, command: str) -> VerifyResult:
        task = asyncio.create_task(
            run_verify(
                command,
                self._workdir,
                timeout_ms=self._verify_timeout_ms,
            )
        )
        self._verify_task = task
        try:
            return await task
        except asyncio.CancelledError:
            return VerifyResult(exit_code=None, cancelled=True)
        finally:
            self._verify_task = None
