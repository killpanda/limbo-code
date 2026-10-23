"""Conversation: one frontend's live session, shared by the TUI and headless.

Sits above the ``TurnPump`` and owns everything a frontend used to wire by
hand, twice:

- **the current Agent + TurnPump pair** — built from config in one place;
  :meth:`switch` loads the target first, so a session that fails to load
  leaves the current one untouched (the old TUI path closed the agent
  first and kept a closed one on failure); ``/new``, ``/fork``,
  ``/handoff`` and the session picker all go through it, and every switch
  reports the new session identity to the integrations;
- **lifecycle reporting** — :meth:`drive` pumps one run's events into a
  frontend callback while reporting ``working`` and the run's outcome
  (``done``/``error``/``idle``) to the integrations;
- **the subagent supervisor** — one per main agent (``[subagents]``):
  the task board, background subagents, notices; frontends subscribe to
  its :class:`~limbo.subagents.events.BackgroundEvent` stream and start a
  main turn on :class:`~limbo.subagents.events.MainWake`;
- **the /goal verify-proposal flow** — after a proposal round,
  :meth:`take_verify_proposal` parses the model's ``<verify_proposal>``
  and the conversation stays in the awaiting state until the frontend
  calls :meth:`accept_verify` or :meth:`skip_verify`.

Busy is ``TurnPump.running`` — the flag the pump releases in the same
await-free section as its last drain, so a submission is either steered
into a queue somebody drains or starts a new turn, never stranded.

Frontends are adapters: they parse input, call these verbs, and render
events and notices. Non-UI layer: never imports textual.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from limbo.agent import Agent
from limbo.config import Config
from limbo.goal import (
    STATUS_ACTIVE,
    GoalState,
    build_initial_prompt,
    build_proposer_prompt,
    parse_verify_proposal,
)
from limbo.integrations.base import CompositeReporter, IntegrationReporter, TurnOutcome
from limbo.learn_trigger import LearnTrigger, user_texts
from limbo.llm.client import LLMClient
from limbo.pump import PumpEvent, TurnPump
from limbo.subagents.events import BackgroundEvent
from limbo.subagents.supervisor import ClientFactory, Supervisor


@dataclass(frozen=True)
class NoVerifyGate:
    """The model answered ``<none/>``: the goal has no objective gate."""


@dataclass(frozen=True)
class VerifyProposals:
    """The model's proposed verify commands (empty: none could be parsed)."""

    commands: list[str]


VerifyProposal = NoVerifyGate | VerifyProposals


class Conversation:
    """The live session a frontend drives (see module docstring)."""

    def __init__(
        self,
        *,
        config: Config,
        workdir: Path,
        llm_client: LLMClient,
        session_dir: Path | None = None,
        resume: Path | None = None,
        reporters: IntegrationReporter | None = None,
        client_factory: ClientFactory | None = None,
        daily: bool = False,
    ) -> None:
        self.config = config
        self.workdir = workdir
        # Replaced by /model (the frontend swaps the client and assigns it).
        self.llm_client = llm_client
        self._session_dir = session_dir
        self._reporters: IntegrationReporter = reporters or CompositeReporter()
        # Subagents get a client of their own from this factory (None:
        # they share the main client — tests and fakes).
        self._client_factory = client_factory
        self._listeners: list[Callable[[BackgroundEvent], None]] = []
        # `limbo daily`: the first new session starts in daily mode; /new
        # then follows whatever mode the current session is in.
        self._agent = self._build_agent(resume, daily=daily)
        self._pump = self._build_pump(self._agent)
        self._supervisor = self._build_supervisor(self._agent)
        # /goal proposal flow: a proposal round is in flight or just ended
        # (parse its output at the next quiet boundary) → the proposals are
        # out and the next answer decides the gate.
        self._proposal_round = False
        self._proposals: list[str] | None = None
        self._learn = self._build_learn_trigger()
        # Frontends turn this off when nobody stays for a background review
        # (headless after stdin closed).
        self.auto_learn = True

    # -- the live pair --------------------------------------------------------

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def pump(self) -> TurnPump:
        return self._pump

    @property
    def busy(self) -> bool:
        """A turn, goal loop, follow-up or /compact owns the agent."""
        return self._pump.running

    def _build_agent(self, resume: Path | None, *, daily: bool = False) -> Agent:
        return Agent(
            config=self.config,
            llm_client=self.llm_client,
            workdir=self.workdir,
            session_dir=self._session_dir,
            resume=resume,
            meta_fields={"daily_mode": True} if daily and resume is None else None,
        )

    def _build_learn_trigger(self) -> LearnTrigger:
        daily = self.config.daily
        return LearnTrigger(daily.learn_after_rounds, daily.learn_cooldown_turns)

    def _build_pump(self, agent: Agent) -> TurnPump:
        return TurnPump(
            agent,
            self.workdir,
            max_rounds=self.config.goal.max_rounds,
            verify_timeout_ms=self.config.goal.verify_timeout_ms,
            after_failed_run=self.config.queue.after_failed_run,
        )

    # -- subagents -----------------------------------------------------------

    @property
    def supervisor(self) -> Supervisor | None:
        """The task board's supervisor (None when ``[subagents]`` is off)."""
        return self._supervisor

    def _build_supervisor(self, agent: Agent) -> Supervisor | None:
        if not self.config.subagents.enabled:
            return None
        supervisor = Supervisor(
            main_agent=agent,
            config=self.config,
            workdir=self.workdir,
            llm_client=self.llm_client,
            client_factory=self._client_factory,
            emit=self._emit_background,
            is_main_busy=lambda: self._pump.running,
        )
        self._pump.before_run = supervisor.flush_notices
        return supervisor

    def subscribe(
        self, listener: Callable[[BackgroundEvent], None]
    ) -> Callable[[], None]:
        """Receive background events (on the event loop); returns unsubscribe."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def _emit_background(self, event: BackgroundEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def start_background(self) -> None:
        """Start scheduling subagents (frontends call it once the loop runs)."""
        if self._supervisor is not None:
            self._supervisor.start()

    def take_wake_input(self) -> str | None:
        """The notices queued for the idle main agent, as one turn input.

        None when there is nothing to deliver or a run already owns the
        agent (its loop-top drain picks them up instead).
        """
        if self._supervisor is None or self.busy:
            return None
        return self._supervisor.take_wake_input()

    # -- session switching ----------------------------------------------------

    def new(self) -> None:
        """Start a fresh session (``/new``)."""
        self._replace(self._build_agent(None, daily=self._agent.daily_mode))

    def switch(self, path: Path) -> None:
        """Resume the session at ``path``.

        The target is loaded before the current agent is closed: if loading
        raises, the current session stays live and usable.
        """
        self._replace(self._build_agent(path))

    def fork(self, keep_turns: int | None = None) -> str:
        """Branch into a new session (``/fork``); returns the parent's id."""
        parent = self._agent.session_id
        self.switch(self._agent.fork(keep_turns))
        return parent

    def handoff(self, note: str) -> tuple[str, str]:
        """Start a linked clean session (``/handoff``).

        Returns ``(parent id, first prompt)``; the frontend runs the prompt
        as the new session's first turn.
        """
        prompt = self._agent.handoff_prompt(note)
        parent = self._agent.session_id
        self.switch(self._agent.handoff(note))
        return parent, prompt

    def _replace(self, agent: Agent) -> None:
        # The old session's subagents stop at their next consistency point
        # and stay "running" on disk: going back to that session resumes
        # them per [subagents] on_resume.
        if self._supervisor is not None:
            self._supervisor.stop()
        self._agent.close()  # release the old session's trace file handle
        self._agent = agent
        # The pump wraps the agent: a new agent always needs a new pump, or
        # it would keep pumping the old history.
        self._pump = self._build_pump(agent)
        self._supervisor = self._build_supervisor(agent)
        self._learn = self._build_learn_trigger()
        self.start_background()
        self._proposal_round = False
        self._proposals = None
        self.report_session()

    def report_session(self) -> None:
        self._reporters.report_session(
            self._agent.session_id, str(self._agent.session_path)
        )

    def close(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
        self._agent.close()

    # -- driving runs ---------------------------------------------------------

    async def drive(
        self,
        events: AsyncIterator[PumpEvent],
        on_event: Callable[[PumpEvent], Awaitable[None] | None],
    ) -> TurnOutcome:
        """Feed one pump run's events to ``on_event``, reporting lifecycle.

        Reports ``working`` first and the run's outcome last. A crash (in
        the pump or in ``on_event``) is reported as ``error`` and re-raised
        for the frontend to render.
        """
        self._reporters.report("working")
        outcome = TurnOutcome()
        seen = len(self._agent.messages)
        try:
            async for event in events:
                outcome.observe(event)
                self._learn.observe(event)
                result = on_event(event)
                if inspect.isawaitable(result):
                    await result
        except Exception as e:
            outcome.fail(f"内部错误：{e}")
            raise
        finally:
            self._reporters.report(outcome.state, message=outcome.message)
        self._maybe_learn(outcome.state, seen)
        return outcome

    # -- /learn: background skill review (daily mode) -------------------------

    def _maybe_learn(self, state: str, seen: int) -> None:
        """After a turn: start an automatic review when the trigger says so."""
        if not self._agent.daily_mode:
            return
        messages = self._agent.messages
        # Compaction shrinks the history mid-turn: then only the last
        # message the user wrote is certain to be this turn's.
        new = messages[seen:] if len(messages) >= seen else messages[-1:]
        for text in user_texts(new):
            self._learn.observe_user_text(text)
        if self._supervisor is not None:
            self._learn.settle(self._supervisor.board.tasks)
        if self._learn.end_turn(state) and self.auto_learn:
            self.learn(auto=True)

    def learn(self, *, auto: bool = False) -> str:
        """Start a skill-reviewer subagent over the conversation so far.

        Returns what to tell the user. ``auto`` (after a busy turn) stays
        quiet when there is nothing to do.
        """
        if not self._agent.daily_mode:
            return "" if auto else "/learn 只在 daily 模式下可用"
        if self._supervisor is None:
            return "" if auto else "/learn 需要子代理（[subagents] enabled = true）"
        if any(t.profile == "skill-reviewer" for t in self._supervisor.board.live()):
            return "" if auto else "已有一个复盘在进行中"
        result = self._supervisor.plan(
            [{
                "title": "复盘：把这次对话的经验写进 skill",
                "brief": (
                    "Review the conversation above and update the daily skills "
                    "as your instructions say."
                ),
                "profile": "skill-reviewer",
                "context": "fork",
            }],
            created_by="user",
        )
        if not result.ids:
            return "" if auto else f"复盘没能开始：{result.message}"
        self._learn.review_started(result.ids[0])
        return f"已在后台开始复盘（{result.ids[0]}），有改动会告诉你"

    # -- /goal: proposal round → confirmation ---------------------------------

    def start_goal(self, text: str) -> str:
        """Set a goal; returns the proposal-round prompt to run."""
        state = self._pump.set_goal(text)
        self._proposal_round = True
        self._proposals = None
        return build_proposer_prompt(state)

    def clear_goal(self) -> bool:
        """Exit goal mode; False when there was no goal."""
        if self._pump.status() is None:
            return False
        self._pump.clear()
        self._proposal_round = False
        self._proposals = None
        return True

    @property
    def awaiting_verify(self) -> bool:
        """Proposals are out; the next answer decides the verify gate."""
        return self._proposals is not None

    @property
    def proposals(self) -> list[str]:
        return list(self._proposals or [])

    def take_verify_proposal(self) -> VerifyProposal | None:
        """At a quiet turn end, read the proposal round's answer.

        None when no proposal round just ended (or the goal no longer
        needs a gate). Otherwise the conversation is awaiting an answer
        (except for :class:`NoVerifyGate`) and reports ``blocked``.
        """
        if not self._proposal_round or self.busy:
            return None
        self._proposal_round = False
        state = self._pump.status()
        if state is None or state.status != STATUS_ACTIVE or state.verify_command:
            return None
        text = next(
            (
                m.content
                for m in reversed(self._agent.messages)
                if m.role == "assistant" and m.content
            ),
            "",
        )
        parsed = parse_verify_proposal(text)
        if parsed is not None and len(parsed) == 0:
            return NoVerifyGate()
        self._proposals = parsed or []
        self._reporters.report("blocked", message="等待确认 /goal 验收命令")
        return VerifyProposals(self.proposals)

    def accept_verify(self, command: str) -> str | None:
        """Adopt ``command`` as the gate; returns the first round's prompt.

        None when the goal went away meanwhile (cleared, switched).
        """
        self._proposals = None
        state = self._pump.status()
        if state is None or state.status != STATUS_ACTIVE:
            self._reporters.report("idle")
            return None
        new_state: GoalState | None = self._pump.set_verify(command)
        if new_state is None:
            self._reporters.report("idle")
            return None
        return build_initial_prompt(new_state)

    def skip_verify(self) -> None:
        """Decline the gate: the goal stays as a single-round task."""
        self._proposals = None
        self._reporters.report("idle")

    def report_blocked(self, message: str) -> None:
        """Re-report ``blocked`` (e.g. the TUI moved on to editing)."""
        self._reporters.report("blocked", message=message)

    def report_idle(self) -> None:
        self._reporters.report("idle")
