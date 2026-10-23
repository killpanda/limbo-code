"""Supervisor: runs the task board for one main session (design §6–§8).

Owned by ``Conversation`` (one per main agent). It

- **folds** the main session's ``task`` records into the board on load
  and applies the ``[subagents] on_resume`` policy to tasks that were
  running when the process stopped;
- **reconciles** after every board change: the pure
  :func:`~limbo.subagents.board.reconcile` decides, this module executes
  (start a runner, make a task ready, cascade a cancel, report a
  deadlock);
- **runs subagents** in-process: each running task is a child ``Agent``
  with its own session file under ``<main id>.agents/`` (compaction,
  steer, interrupts and crash recovery come with it), driven by an
  asyncio task here;
- **delivers notices** to the main agent through its steer queue
  (``source="subagent"``), coalesced into one ``<subagent_updates>``
  message; queue journaling plus the board's ``reported`` set make the
  delivery at-least-once with dedupe across a crash;
- exposes the **verbs** behind the task tools, the ``/task`` commands and
  ``@name`` routing — every control path ends in the same methods.

Non-UI layer: never imports textual.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from limbo.agent import ErrorEvent, TextDelta, ToolCallRequest, ToolResultEvent, UsageUpdate
from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.sessions import save_session
from limbo.steer import FOLLOW_UP, STEER
from limbo.subagents import worktree as wt
from limbo.subagents.board import (
    TERMINAL,
    BoardError,
    CancelTask,
    Deadlock,
    Finalize,
    MakeReady,
    Outcome,
    Start,
    Task,
    TaskBoard,
    new_notice_id,
    reconcile,
)
from limbo.subagents.events import (
    DELIVERED_KINDS,
    BackgroundEvent,
    MainWake,
    Notice,
    NoticeEvent,
    PlanProposed,
    TaskChanged,
)
from limbo.subagents.hooks import SUBAGENT_TOOL_NAMES, ClaimHook, NoEditHook, ReadOnlyHook
from limbo.subagents.profiles import (
    NO_EDIT,
    READ_ONLY,
    Profile,
    discover_profiles,
    format_profiles_for_prompt,
    resolve_profile,
)
from limbo.subagents.project_scan import ProjectFacts, format_facts, scan_project
from limbo.subagents.tools import main_tools, subagent_tools

if TYPE_CHECKING:
    from limbo.agent import Agent

ClientFactory = Callable[[Config], LLMClient]

STATUS_LABEL = {
    "proposed": "待确认",
    "pending": "排队",
    "ready": "就绪",
    "running": "运行中",
    "waiting": "等待",
    "paused": "已暂停",
    "completing": "收尾中",
    "done": "完成",
    "failed": "失败",
    "cancelled": "已取消",
}

MAIN_PROMPT = """## Subagents

You can delegate work to background subagents with plan_tasks. They run in
parallel with their own context while you keep working with the user; their
results, questions and failures come back to you as <subagent_updates>
messages. Manage them with task_status, task_send, answer and task_control.

Delegate when:
- two or more independent parts of the code must change;
- a broad search or reading would flood your context and you only need the
  conclusion (use the explore profile);
- an independent review or a long verification can run beside the work.
Do not delegate small single-file changes, strictly sequential steps where
every step needs the previous result, work that needs most of this
conversation's context, or anything the user is actively discussing with you.

Give every task a self-contained brief (subagents cannot see this
conversation), declare scope_paths for tasks that write files, and express
ordering with depends_on. When the user asks to pause, resume, cancel, retry,
reprioritize or redirect a subagent, use task_control / task_send. Answer
subagent questions with answer; ask the user when the decision is theirs."""

MAIN_ONLY_TOOLS = ("memory", "ask_user", "skill_manage", "session_search")

CHILD_PROMPT = """<subagent_role>
You are the subagent "{agent}" of limbo, working on task {id}: {title}.
The main agent coordinates the work; you cannot see its conversation and the
user does not watch you live.
- Call checkpoint(phase, summary) when you move to a new phase.
- If your task turns out to depend on another task's result, call
  need(on=[task ids], reason). If it needs work nobody owns yet, call
  need(new={{title, brief}}, reason). Do not do other tasks' work, and do
  not edit paths another task claimed.
- If a decision belongs to the main agent or the user, call ask_main.
- When you are done, call finish(summary, files_changed); the summary is all
  the main agent reads. If you cannot complete the task, call
  finish(status="failed") and say why.
{scope}{others}</subagent_role>"""

# A cache-parity fork child (profile inherit_prompt) keeps the main
# session's system prompt and tool list byte for byte, so its role travels
# in the first user message instead, and it has no task tools of its own:
# its final plain-text reply is the result.
FORK_CHILD_PROMPT = """<subagent_role>
From here on you are the subagent "{agent}" of limbo, working on task {id}:
{title}. The conversation above is the main session's, shown for context;
the user does not watch you live.
- The tool list is the main session's, but you may only call: {tools}.
  Any other tool fails.
- When you are done, reply with your result as plain text and no tool
  calls: that reply is all the main agent reads.
</subagent_role>"""


@dataclass
class PlanResult:
    message: str
    proposed: bool = False
    ids: list[str] = field(default_factory=list)


@dataclass
class _Runner:
    task_id: str
    agent: Agent | None = None
    future: asyncio.Task[None] | None = None
    calls: int = 0
    last_text: str = ""
    last_error: str = ""
    over_budget: bool = False
    budget_reason: str = ""
    # First line of every successful state-changing call (named in the
    # outcome when the budget ends the task before it reports).
    writes: list[str] = field(default_factory=list)
    resumed: bool = False


class Supervisor:
    """The task board's scheduler and the subagents' owner (see module doc)."""

    def __init__(
        self,
        *,
        main_agent: Agent,
        config: Config,
        workdir: Path,
        llm_client: LLMClient,
        client_factory: ClientFactory | None = None,
        emit: Callable[[BackgroundEvent], None] | None = None,
        is_main_busy: Callable[[], bool] | None = None,
        profiles: dict[str, Profile] | None = None,
        facts: ProjectFacts | None = None,
    ) -> None:
        self.main = main_agent
        self.config = config
        self.settings = config.subagents
        self.workdir = workdir
        self.llm_client = llm_client
        self._client_factory = client_factory
        self._emit_cb = emit
        self.is_main_busy = is_main_busy or (lambda: False)
        self.profiles = profiles if profiles is not None else discover_profiles(workdir)
        self._facts = facts
        self.board = TaskBoard()
        self._runners: dict[str, _Runner] = {}
        self._pending: list[Notice] = []
        self._deadlocks: set[str] = set()
        self._activity: dict[str, str] = {}
        self._tokens: dict[str, int] = {}
        self._stopped = False
        self._scheduled = False
        self.interrupted_on_load: list[str] = []
        self._attach()

    # -- setup -----------------------------------------------------------------

    @property
    def facts(self) -> ProjectFacts:
        if self._facts is None:
            self._facts = scan_project(self.workdir)
        return self._facts

    @property
    def agents_dir(self) -> Path:
        path = self.main.session_path
        return path.parent / f"{path.stem}.agents"

    def _attach(self) -> None:
        for rec in self.main.loaded_task_records:
            self.board.apply(rec)
        self.board.set_sink(self._on_record)
        self.main.journal.task_snapshot = lambda: (
            self.board.snapshot() if self.board.tasks else None
        )
        # Notices restored in the main queue were delivered (dedupe).
        refs = [
            r for item in self.main.queued_items if item.source == "subagent" for r in item.refs
        ]
        if refs:
            self.board.mark_reported(refs)
        self._recover_running()
        self._regenerate_notices()
        if self.settings.auto != "off":
            self.main.install(
                tools=main_tools(self.workdir, self),
                prompt_section=self._main_prompt(),
            )

    def _main_prompt(self) -> str:
        return "\n\n".join(
            [
                MAIN_PROMPT,
                format_profiles_for_prompt(self.profiles),
                format_facts(self.facts),
            ]
        )

    def _recover_running(self) -> None:
        """Tasks that were running when the process stopped (design §8.4)."""
        policy = self.settings.on_resume
        for task in self.board.with_status("running"):
            self.interrupted_on_load.append(task.id)
            note = "（进程重启，你的工作被中断）先检查工作区现状，再从上次打点继续。"
            if policy == "auto":
                self.board.update(
                    task.id, status="ready", resume_note=_join(task.resume_note, note)
                )
            else:
                self.board.update(
                    task.id,
                    status="paused",
                    paused_from="running",
                    paused_reason="进程重启时中断",
                    resume_note=_join(task.resume_note, note),
                )
        if self.interrupted_on_load and policy != "auto":
            ids = "、".join(self.interrupted_on_load)
            self._notice(
                "",
                "progress",
                "",
                headline=(
                    f"{len(self.interrupted_on_load)} 个子任务在上次退出时中断（{ids}），"
                    "已暂停；/task resume all 继续"
                ),
            )

    def _regenerate_notices(self) -> None:
        for task in self.board.tasks.values():
            if task.terminal and task.outcome is not None and task.created_by != "user":
                nid = _result_id(task)
                if nid not in self.board.reported and task.outcome.status != "cancelled":
                    self._pending.append(self._result_notice(task))
            q = task.open_question()
            if q is not None and q.id not in self.board.reported:
                self._pending.append(self._question_notice(task, q.id, q.text))

    # -- events and scheduling -------------------------------------------------

    def _emit(self, event: BackgroundEvent) -> None:
        if self._emit_cb is not None:
            try:
                self._emit_cb(event)
            except Exception:  # noqa: BLE001 - a frontend bug must not stop scheduling
                self.main.trace.log("subagent_emit_error", error=traceback.format_exc())

    def _on_record(self, rec: dict[str, Any]) -> None:
        try:
            self.main.journal.task_event(rec)
        except OSError as e:
            self.main.trace.log("session_save_error", phase="task", error=str(e))
        tid = rec.get("id") or (rec.get("task") or {}).get("id")
        if rec.get("op") in ("created", "update"):
            self.main.trace.log(
                "task_event",
                op=rec.get("op"),
                id=tid,
                fields=sorted((rec.get("fields") or {}).keys()),
                status=(rec.get("fields") or rec.get("task") or {}).get("status"),
            )
        if isinstance(tid, str) and tid in self.board.tasks:
            self._emit(TaskChanged(self.board.tasks[tid], self._activity.get(tid, "")))
        self.kick()

    def kick(self) -> None:
        """Schedule a reconcile on the running loop (no-op without one)."""
        if self._stopped or self._scheduled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._scheduled = True
        loop.call_soon(self._reconcile)

    def start(self) -> None:
        """Begin scheduling (frontends call it once the loop runs)."""
        self.kick()

    def _reconcile(self) -> None:
        self._scheduled = False
        if self._stopped:
            return
        for _ in range(200):
            actions = [
                a
                for a in reconcile(self.board, max_parallel=self.settings.max_parallel)
                if not (isinstance(a, Start) and a.task_id in self._runners)
                and not (isinstance(a, Deadlock) and a.signature in self._deadlocks)
            ]
            if not actions:
                break
            for action in actions:
                try:
                    self._execute(action)
                except BoardError as e:
                    self.main.trace.log("subagent_reconcile_error", error=str(e))
        self._scheduled = False
        self._deliver()

    def _execute(self, action: Any) -> None:
        if isinstance(action, Start):
            self._start(action.task_id)
        elif isinstance(action, MakeReady):
            task = self.board.get(action.task_id)
            self.board.update(
                task.id,
                status="ready",
                waiting_on=[],
                wait_reason="",
                resume_note=_join(task.resume_note, action.note),
            )
        elif isinstance(action, CancelTask):
            self._cancel(action.task_id, action.reason, by="system")
        elif isinstance(action, Finalize):
            task = self.board.get(action.task_id)
            status = task.outcome.status if task.outcome else "done"
            self.board.update(task.id, status=status)
        elif isinstance(action, Deadlock):
            self._deadlocks.add(action.signature)
            self._notice(
                "",
                "alert",
                "子任务互相等待，没有任何任务能继续：\n"
                + "\n".join(action.chain)
                + "\n请取消其中一个或用 task_control(edit) 调整依赖。",
                headline="子任务死锁：" + "；".join(action.chain),
            )

    # -- runners ---------------------------------------------------------------

    def _start(self, task_id: str) -> None:
        task = self.board.get(task_id)
        path = Path(task.session_file) if task.session_file else self._session_path(task)
        runner = _Runner(task_id)
        self._runners[task_id] = runner
        self.board.set_status(task_id, "running", session_file=str(path), paused_reason="")
        runner.future = asyncio.ensure_future(self._run(runner))

    def _session_path(self, task: Task) -> Path:
        return self.agents_dir / f"{task.id}-{task.agent_name}.jsonl"

    async def _run(self, runner: _Runner) -> None:
        task_id = runner.task_id
        try:
            task = self.board.get(task_id)
            if task.isolation == "worktree" and not (
                task.worktree and Path(task.worktree).is_dir()
            ):
                tree = await asyncio.to_thread(
                    wt.create, self.workdir, self.main.session_id, task.id
                )
                self.board.update(task_id, worktree=str(tree.path))
            agent, resumed = self._open_agent(self.board.get(task_id))
            runner.agent = agent
            runner.resumed = resumed
            text = self._first_input(self.board.get(task_id), resumed)
            self.board.update(task_id, resume_note="", inbox=[])
            while True:
                async for event in agent.run(text):
                    self._observe(runner, event)
                if runner.over_budget:
                    self._finish_over_budget(runner)
                    break
                if self.board.get(task_id).status != "running":
                    break
                if agent.was_interrupted:
                    break
                leftovers = agent.drain_steer()
                if leftovers:
                    text = "\n\n".join(item.text for item in leftovers)
                    continue
                if agent.last_run_failed:
                    self._finish(
                        task_id,
                        summary=runner.last_text,
                        failed=True,
                        reason=runner.last_error or "LLM 调用失败",
                    )
                    break
                # The turn ended without finish(): its last answer is the
                # result (models often just answer).
                self._finish(
                    task_id,
                    summary=_last_answer(agent) or "（子代理没有给出总结）",
                    auto=True,
                )
                break
        except asyncio.CancelledError:
            # stop(): the task stays "running" on disk and is recovered on
            # the next load (on_resume policy).
            raise
        except Exception as e:  # noqa: BLE001
            self.main.trace.log("subagent_crash", task=task_id, error=traceback.format_exc())
            with contextlib.suppress(BoardError):
                if not self.board.get(task_id).terminal:
                    self._finish(task_id, summary="", failed=True, reason=f"子代理异常：{e}")
        finally:
            self._runners.pop(task_id, None)
            agent_ = runner.agent
            if agent_ is not None:
                with contextlib.suppress(BoardError):
                    self.board.record_usage(task_id, agent_.session_total_tokens)
                agent_.close()
            self._activity.pop(task_id, None)
            self.kick()

    def _observe(self, runner: _Runner, event: Any) -> None:
        task_id = runner.task_id
        if isinstance(event, ToolCallRequest):
            runner.calls += 1
            self._set_activity(task_id, f"⚙ {event.name}")
            every = self.settings.auto_checkpoint_every
            if every and runner.calls % every == 0:
                with contextlib.suppress(BoardError):
                    self.board.add_checkpoint(
                        task_id,
                        f"已执行 {runner.calls} 次工具调用",
                        runner.last_text[-200:],
                        auto=True,
                    )
        elif isinstance(event, TextDelta):
            runner.last_text = (runner.last_text + event.text)[-2000:]
            if self._activity.get(task_id) != "✎ 回复中":
                self._set_activity(task_id, "✎ 回复中")
        elif isinstance(event, ErrorEvent):
            runner.last_error = event.message
        elif isinstance(event, ToolResultEvent):
            agent = runner.agent
            if (
                event.result.success
                and event.name not in SUBAGENT_TOOL_NAMES
                and agent is not None
                and agent.registry.replay_policy(event.name) != "safe"
            ):
                first = (event.result.output or "").strip().splitlines()
                runner.writes.append(f"{event.name}：{first[0][:160] if first else '完成'}")
        elif isinstance(event, UsageUpdate):
            self._tokens[task_id] = event.total_tokens
            task = self.board.tasks.get(task_id)
            budget = task.max_tokens if task is not None else 0
            # The budget counts fresh tokens: a fork child re-sends the
            # whole inherited conversation every iteration, and those cache
            # hits would otherwise exhaust it in a handful of requests.
            if budget and event.fresh_tokens > budget and not runner.over_budget:
                runner.over_budget = True
                runner.budget_reason = (
                    f"超出 token 预算（新增 {event.fresh_tokens} > {budget}，缓存命中不计）"
                )
                # Not interrupt(): the response that crossed the line and
                # its tool batch complete (their writes land), then the
                # turn ends before the next request.
                if runner.agent is not None:
                    runner.agent.request_stop(runner.budget_reason)

    def _finish_over_budget(self, runner: _Runner) -> None:
        """The budget ended the turn: report what the task got done."""
        task = self.board.tasks.get(runner.task_id)
        if task is None or task.terminal:
            return  # it called finish in the crossing batch
        lines = [runner.budget_reason + "，已在下一次请求前收尾，结果可能不完整"]
        if runner.writes:
            lines.append("已执行的改动：\n" + "\n".join(f"- {w}" for w in runner.writes[-20:]))
        text = runner.last_text.strip()
        self._finish(
            runner.task_id,
            summary=text,
            failed=not (text or runner.writes),
            reason="\n".join(lines),
        )

    def _set_activity(self, task_id: str, text: str) -> None:
        self._activity[task_id] = text
        task = self.board.tasks.get(task_id)
        if task is not None:
            self._emit(TaskChanged(task, text))

    def tokens(self, task_id: str) -> int:
        task = self.board.tasks.get(task_id)
        return self._tokens.get(task_id) or (task.tokens if task else 0)

    def activity(self, task_id: str) -> str:
        return self._activity.get(task_id, "")

    def _model_for(self, profile: Profile) -> str:
        if profile.model in ("", "inherit"):
            return self.config.llm.model
        if profile.model == "cheap":
            return self.settings.cheap_model or self.config.llm.model
        return profile.model

    def _client_for(self, cfg: Config) -> LLMClient:
        """A client of its own per subagent (clients carry per-session state
        such as the prompt cache key); the main client when none can be
        built."""
        factory = self._client_factory
        if factory is None:
            return self.llm_client
        try:
            return factory(cfg)
        except Exception:  # noqa: BLE001 - missing key etc.: share the main client
            cfg.llm.model = self.config.llm.model
            return self.llm_client

    def _open_agent(self, task: Task) -> tuple[Agent, bool]:
        """The task's child agent, and whether it resumed its own history."""
        from limbo.agent import Agent

        profile = resolve_profile(self.profiles, task.profile)
        cfg = self.config.model_copy(deep=True)
        cfg.llm.max_iterations = profile.max_iterations
        cfg.llm.model = self._model_for(profile)
        client = self._client_for(cfg)
        workdir = Path(task.worktree) if task.worktree else self.workdir
        path = Path(task.session_file) if task.session_file else self._session_path(task)
        resumed = path.exists()
        if not resumed and task.context == "fork":
            self._write_fork(path, task)
        if path.exists():
            agent = Agent(
                config=cfg, llm_client=client, workdir=workdir, session_dir=path.parent, resume=path
            )
        else:
            agent = Agent(
                config=cfg,
                llm_client=client,
                workdir=workdir,
                session_dir=path.parent,
                session_file=path,
                meta_fields={
                    "role": "subagent",
                    "task_id": task.id,
                    "parent_session": self.main.session_id,
                    "title": f"{task.id} · {task.title}",
                    # Daily children get web_fetch and the daily skills too.
                    "daily_mode": self.main.daily_mode,
                },
            )
        self._configure(agent, task, profile, workdir)
        return agent, resumed

    def _inherits_prompt(self, task: Task, profile: Profile) -> bool:
        """Cache parity (hermes' background review): a fork child on the
        main model sends the main session's exact prefix, so even its first
        request is a prompt-cache hit. On another model the cache is cold
        anyway and the child keeps its own prompt and task tools."""
        return (
            task.context == "fork"
            and profile.inherit_prompt
            and self._model_for(profile) == self.main.config.llm.model
        )

    def _write_fork(self, path: Path, task: Task) -> None:
        """context=fork: the child starts from the main history up to the
        current turn's user message (never a half-finished tool batch)."""
        from limbo.sessions import SessionMeta

        messages = list(self.main.messages)
        turns = [i for i, m in enumerate(messages) if i > 0 and m.role == "user"]
        cut = turns[-1] + 1 if turns else 1
        # An idle main agent's history is complete (a skill review needs the
        # final answer too); mid-turn, stop before the half-finished turn.
        kept = messages if not self.is_main_busy() else messages[:cut]
        meta = SessionMeta(
            id=path.stem,
            workdir=str(self.workdir.resolve()),
            model=self.config.llm.model,
            title=f"{task.id} · {task.title}",
            role="subagent",
            task_id=task.id,
            parent_session=self.main.session_id,
            fork_at=kept[-1].id if len(kept) > 1 else None,
            tool_intents=True,
            daily_mode=self.main.daily_mode,
        )
        save_session(path, meta, kept)

    def _configure(self, agent: Agent, task: Task, profile: Profile, workdir: Path) -> None:
        hooks: list[Any] = []
        remove: list[str] = []
        names = profile.tool_names
        if names is not None:
            remove = [n for n in agent.registry.names() if n not in names]
        else:
            # Talking to the user and curating memory/skills stay with the
            # main agent unless a profile lists them explicitly.
            remove = [n for n in MAIN_ONLY_TOOLS if agent.registry.get(n) is not None]
        if profile.tools == READ_ONLY:
            registry = agent.registry
            hooks.append(ReadOnlyHook(lambda n: registry.replay_policy(n) == "safe"))
        elif profile.tools == NO_EDIT:
            hooks.append(NoEditHook())
        if profile.writes and task.isolation == "shared":
            hooks.append(ClaimHook(task.id, self.board, workdir))
        if self._inherits_prompt(task, profile):
            # Advertising is not permission: the main tool list is shown,
            # the child's trimmed registry and hooks decide what runs.
            agent.install(
                hooks=hooks,
                remove_tools=remove,
                system_prompt=self.main.system_prompt,
                advertised_tools=self.main.tool_definitions(),
            )
            if hasattr(agent.llm_client, "prompt_cache_key"):
                agent.llm_client.prompt_cache_key = self.main.session_id
            return
        agent.install(
            tools=subagent_tools(workdir, self, task.id),
            hooks=hooks,
            remove_tools=remove,
            prompt_section=self._child_prompt(task, profile),
        )

    def _child_prompt(self, task: Task, profile: Profile) -> str:
        scope = ""
        if task.scope_paths:
            scope = f"- Your task claims these paths: {', '.join(task.scope_paths)}\n"
        others = [
            f"  {t.id} ({t.agent_name}, {t.status}): {t.title}"
            + (f" — claims {', '.join(t.scope_paths)}" if t.scope_paths else "")
            for t in self.board.live()
            if t.id != task.id
        ]
        other_text = (
            ("- Other tasks on the board:\n" + "\n".join(others[:20]) + "\n") if others else ""
        )
        prompt = CHILD_PROMPT.format(
            agent=task.agent_name, id=task.id, title=task.title, scope=scope, others=other_text
        )
        return prompt + ("\n\n" + profile.body if profile.body else "")

    def _first_input(self, task: Task, resumed: bool) -> str:
        inbox = "\n".join(f"- {m}" for m in task.inbox)
        if resumed:
            parts = ["（任务继续）" + (task.resume_note or "继续你的任务。")]
            cp = task.last_checkpoint
            if cp is not None:
                parts.append(
                    f"你上次的打点：{cp.phase}" + (f" — {cp.summary}" if cp.summary else "")
                )
            if inbox:
                parts.append("主 agent / 用户补充：\n" + inbox)
            return "\n\n".join(parts)
        parts = [f"任务 {task.id}：{task.title}", task.brief.strip()]
        profile = resolve_profile(self.profiles, task.profile)
        if self._inherits_prompt(task, profile):
            names = profile.tool_names or []
            role = FORK_CHILD_PROMPT.format(
                agent=task.agent_name,
                id=task.id,
                title=task.title,
                tools=", ".join(names) or "（主会话的全部工具）",
            )
            parts = [role, profile.body, *parts]
        if task.acceptance:
            parts.append(f"验收标准：{task.acceptance}")
        if task.scope_paths:
            parts.append(f"你认领的路径：{', '.join(task.scope_paths)}")
        if task.resume_note:
            parts.append("前置信息：\n" + task.resume_note)
        if inbox:
            parts.append("主 agent / 用户补充：\n" + inbox)
        return "\n\n".join(p for p in parts if p)

    # -- shutdown --------------------------------------------------------------

    def stop(self) -> None:
        """Stop every subagent without writing outcomes (the next load
        recovers them). Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True
        for runner in list(self._runners.values()):
            if runner.agent is not None:
                runner.agent.interrupt("limbo 退出，任务会在下次启动时恢复")
            if runner.future is not None:
                runner.future.cancel()
        self.board.set_sink(None)
        self.main.journal.task_snapshot = None

    @property
    def working(self) -> bool:
        """Subagents are running or about to (paused, proposed and
        question-blocked work waits on a person and does not count)."""
        return bool(self._runners) or bool(self.board.with_status("ready", "running", "completing"))

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Test/CLI helper: wait until no subagent runs and nothing is ready."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
            if self._runners or self._scheduled:
                continue
            if self.board.with_status("ready"):
                continue
            return
        raise TimeoutError("subagents still running")

    # -- notices ---------------------------------------------------------------

    def _notice(
        self, task_id: str, kind: str, text: str, *, headline: str = "", nid: str | None = None
    ) -> Notice:
        notice = Notice(
            id=nid or new_notice_id(f"{task_id or 'board'}:{kind}"),
            task_id=task_id,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            headline=headline or text.splitlines()[0][:160] if (headline or text) else "",
        )
        if kind in DELIVERED_KINDS:
            self._pending.append(notice)
        self._emit(NoticeEvent(notice))
        self.kick()
        return notice

    def _result_notice(self, task: Task) -> Notice:
        oc = task.outcome
        assert oc is not None
        kind = "result" if oc.status == "done" else ("alert" if oc.status == "failed" else "info")
        limit = self.settings.result_chars
        body = oc.summary.strip()
        if len(body) > limit:
            body = body[:limit] + "…（已截断，完整内容见记录）"
        lines = [body] if body else []
        if oc.reason:
            lines.append(f"原因：{oc.reason}")
        if oc.files_changed:
            lines.append("改动文件：" + ", ".join(oc.files_changed[:30]))
        if oc.follow_ups:
            lines.append("后续建议：" + "；".join(oc.follow_ups[:10]))
        if task.worktree:
            lines.append(
                f'worktree：{task.worktree}（合并：task_control(id="{task.id}", action="merge")）'
            )
        if task.session_file:
            lines.append(f"完整记录：{task.session_file}")
        text = (
            f'<subagent_update task="{task.id}" agent="{task.agent_name}" kind="{kind}" '
            f'status="{oc.status}">\n' + "\n".join(lines) + "\n</subagent_update>"
        )
        label = STATUS_LABEL.get(oc.status, oc.status)
        first = (oc.reason or body).splitlines()
        headline = f"{task.id} {task.agent_name} {label}：{first[0][:120] if first else task.title}"
        return Notice(_result_id(task), task.id, kind, text, headline)  # type: ignore[arg-type]

    def _question_notice(self, task: Task, qid: str, text: str) -> Notice:
        body = (
            f'<subagent_update task="{task.id}" agent="{task.agent_name}" '
            f'kind="question" id="{qid}">\n'
            f'{text}\n（回答：answer(question_id="{qid}", text=...)；需要用户决定时先问用户）\n'
            "</subagent_update>"
        )
        return Notice(
            qid, task.id, "question", body, f"{task.id} {task.agent_name} 提问：{text[:120]}"
        )

    def _push(self, notice: Notice) -> None:
        self._pending.append(notice)
        self._emit(NoticeEvent(notice))
        self.kick()

    def pending_notices(self) -> list[Notice]:
        return [n for n in self._pending if n.id not in self.board.reported]

    def _deliver(self) -> None:
        deliverable = [
            n
            for n in self._pending
            if n.kind in DELIVERED_KINDS and n.id not in self.board.reported
        ]
        self._pending = deliverable
        if not deliverable:
            return
        if self.is_main_busy():
            self._queue(deliverable, wake=False)
            return
        wake = set(self.settings.wake_main_on)
        kinds = {n.kind for n in deliverable}
        urgent = bool(kinds & {"question", "alert"} & wake)
        wave_done = not self.board.with_status("running", "ready")
        results = "result" in kinds and ("result" in wake or ("wave_done" in wake and wave_done))
        if urgent or results:
            self._queue(deliverable, wake=True)

    def flush_notices(self) -> None:
        """Queue every undelivered notice now (before a user turn starts, so
        the main agent sees them ahead of the user's message)."""
        deliverable = [
            n
            for n in self._pending
            if n.kind in DELIVERED_KINDS and n.id not in self.board.reported
        ]
        if deliverable:
            self._queue(deliverable, wake=False)

    def _queue(self, notices: list[Notice], *, wake: bool) -> None:
        board_line = " · ".join(
            f"{t.id} {t.status}" for t in self.board.tasks.values() if not t.terminal
        )
        text = (
            f'<subagent_updates board="{board_line or "no live tasks"}">\n'
            + "\n".join(n.text for n in notices)
            + "\n</subagent_updates>"
        )
        ids = tuple(n.id for n in notices)
        self.main.steer(text, mode=STEER, source="subagent", refs=ids)
        self.board.mark_reported(ids)
        self._pending = [n for n in self._pending if n.id not in ids]
        if wake:
            self._emit(MainWake())

    def take_wake_input(self) -> str | None:
        """Drain the notices queued for the idle main agent into one input."""
        if not any(i.source == "subagent" for i in self.main.queued_items):
            return None
        drained = self.main.drain_steer((STEER,))
        return "\n\n".join(i.text for i in drained) or None

    # -- verbs: main agent / user ------------------------------------------------

    def plan(self, specs: list[dict[str, Any]], *, created_by: str = "main") -> PlanResult:
        if len(self.board.live()) + len(specs) > self.settings.max_tasks:
            return PlanResult(
                f"超出任务上限：最多 {self.settings.max_tasks} 个未结束任务，"
                f"当前 {len(self.board.live())} 个。合并任务或等现有任务结束。"
            )
        ordered = _topo(specs)
        if isinstance(ordered, str):
            return PlanResult(ordered)
        status = (
            "proposed" if self.settings.auto == "suggest" and created_by != "user" else "pending"
        )
        keys: dict[str, str] = {}
        created: list[Task] = []
        git = None
        for spec in ordered:
            deps: list[str] = []
            for dep in spec.get("depends_on") or []:
                dep = str(dep)
                if dep in keys:
                    deps.append(keys[dep])
                else:
                    found = self.board.find(dep)
                    if found is None:
                        return PlanResult(f"依赖 {dep} 既不是本计划的 key，也不是已有任务")
                    deps.append(found.id)
            profile = resolve_profile(self.profiles, str(spec.get("profile") or "worker"))
            isolation = str(
                spec.get("isolation") or profile.isolation or self.settings.default_isolation
            )
            if isolation == "worktree":
                if git is None:
                    git = wt.is_git_repo(self.workdir)
                if not git:
                    isolation = "shared"
            scope = spec.get("scope_paths") or list(profile.scope_paths)
            task = self.board.create(
                title=str(spec.get("title") or "untitled"),
                brief=str(spec.get("brief") or ""),
                profile=profile.name,
                depends_on=deps,
                status=status,  # type: ignore[arg-type]
                created_by=created_by,  # type: ignore[arg-type]
                acceptance=str(spec.get("acceptance") or ""),
                scope_paths=[str(s) for s in scope] if isinstance(scope, list) else [str(scope)],
                priority=int(spec.get("priority") or 0),
                isolation="worktree" if isolation == "worktree" else "shared",
                context="fork"
                if spec.get("context") == "fork" or profile.context == "fork"
                else "brief",
                on_dep_fail=self.settings.on_dep_fail,
                max_tokens=self.settings.task_token_budget,
            )
            created.append(task)
            if spec.get("key"):
                keys[str(spec["key"])] = task.id
        listing = "\n".join(
            f"- {t.id} [{t.profile}] {t.title}"
            + (f"（依赖 {', '.join(t.depends_on)}）" if t.depends_on else "")
            for t in created
        )
        ids = [t.id for t in created]
        if status == "proposed":
            self._emit(PlanProposed(ids))
            return PlanResult(
                f"计划已提交给用户确认：\n{listing}\n用户确认前任务不会开始。"
                "现在结束本轮，简短告诉用户计划的要点即可。",
                proposed=True,
                ids=ids,
            )
        return PlanResult(
            f"已创建 {len(created)} 个任务并在后台开始：\n{listing}\n"
            "结果会以 <subagent_updates> 回到你这里；现在可以先回复用户。",
            ids=ids,
        )

    def _resolve(self, ref: str, statuses: tuple[str, ...] | None = None) -> list[Task]:
        ref = ref.strip()
        if ref == "all":
            tasks = self.board.live()
        else:
            task = self.board.find(ref)
            if task is None:
                raise BoardError(f"没有任务或子代理 {ref}")
            tasks = [task]
        if statuses is not None and ref == "all":
            tasks = [t for t in tasks if t.status in statuses]
        return tasks

    def control(
        self,
        ref: str,
        action: str,
        *,
        note: str = "",
        priority: Any = None,
        fresh: bool = False,
        brief: Any = None,
        depends_on: list[str] | None = None,
        by: str = "main",
    ) -> str:
        try:
            if action == "pause":
                out = [
                    self._pause(t, note)
                    for t in self._resolve(ref, ("running", "waiting", "ready", "pending"))
                ]
            elif action == "resume":
                out = [self._resume(t, note) for t in self._resolve(ref, ("paused",))]
            elif action == "cancel":
                out = [
                    self._cancel(
                        t.id, note or ("用户取消" if by == "user" else "主 agent 取消"), by=by
                    )
                    for t in self._resolve(ref)
                ]
            elif action == "retry":
                out = [self._retry(t, note, fresh) for t in self._resolve(ref)]
            elif action == "reprioritize":
                out = []
                for t in self._resolve(ref):
                    self.board.update(t.id, priority=int(priority or 0))
                    out.append(f"{t.id} 优先级 → {int(priority or 0)}")
            elif action == "edit":
                out = [self._edit(t, brief, depends_on, note) for t in self._resolve(ref)]
            elif action == "approve":
                out = [self._approve(t) for t in self._resolve(ref, ("proposed",))]
            elif action == "reject":
                out = [self._reject(t, note) for t in self._resolve(ref, ("proposed",))]
            elif action == "merge":
                out = [self._merge(t) for t in self._resolve(ref)]
            else:
                return f"未知动作 {action}"
        except BoardError as e:
            return str(e)
        if not out:
            return "没有符合条件的任务"
        message = "\n".join(out)
        if by == "user":
            self._notice(
                "",
                "info",
                f'<subagent_update kind="info">用户直接操作了任务（/task {action} {ref}）：'
                f"{message}</subagent_update>",
                headline=message,
            )
        return message

    def _pause(self, task: Task, note: str) -> str:
        if task.status not in ("running", "waiting", "ready", "pending"):
            return f"{task.id} 当前{STATUS_LABEL.get(task.status)}，不能暂停"
        self.board.update(
            task.id,
            status="paused",
            paused_from=task.status,
            paused_reason=note or "已暂停",
        )
        runner = self._runners.get(task.id)
        if runner is not None and runner.agent is not None:
            runner.agent.interrupt("任务被暂停")
        return f"{task.id} 已暂停"

    def _resume(self, task: Task, note: str) -> str:
        if task.status != "paused":
            return f"{task.id} 不在暂停状态"
        back = task.paused_from if task.paused_from in ("waiting", "pending") else "ready"
        self.board.update(
            task.id,
            status=back,
            paused_from="",
            paused_reason="",
            resume_note=_join(task.resume_note, note),
        )
        return f"{task.id} 已恢复"

    def _cancel(self, task_id: str, reason: str, *, by: str) -> str:
        task = self.board.get(task_id)
        if task.terminal:
            return f"{task.id} 已经结束（{STATUS_LABEL.get(task.status)}）"
        self.board.update(
            task.id,
            status="cancelled",
            outcome=Outcome(status="cancelled", reason=reason),
        )
        runner = self._runners.get(task.id)
        if runner is not None and runner.agent is not None:
            runner.agent.interrupt("任务被取消")
        if by == "system":
            self._push(self._result_notice(self.board.get(task.id)))
        return f"{task.id} 已取消"

    def _retry(self, task: Task, note: str, fresh: bool) -> str:
        if task.status not in ("failed", "cancelled"):
            return f"{task.id} 当前{STATUS_LABEL.get(task.status)}，只有失败或取消的任务能重试"
        previous = task.outcome
        prior = ""
        if previous is not None:
            prior = f"上一次尝试{STATUS_LABEL.get(previous.status)}" + (
                f"：{previous.reason}" if previous.reason else ""
            )
            if previous.summary:
                prior += f"\n上次的总结：{previous.summary[:1000]}"
        fields: dict[str, Any] = {
            "status": "ready",
            "outcome": None,
            "attempt": task.attempt + 1,
            "resume_note": _join(prior, note),
        }
        if fresh:
            fields["session_file"] = None
        self.board.update(task.id, **fields)
        return f"{task.id} 已重新排队（第 {task.attempt + 2} 次尝试）"

    def _edit(self, task: Task, brief: Any, depends_on: list[str] | None, note: str) -> str:
        if task.status in ("proposed", "pending", "ready", "paused") and not (
            task.status == "ready" and task.session_file
        ):
            fields: dict[str, Any] = {}
            if isinstance(brief, str) and brief.strip():
                fields["brief"] = brief
            if depends_on is not None:
                ids = []
                for ref in depends_on:
                    found = self.board.find(ref)
                    if found is None:
                        raise BoardError(f"依赖 {ref} 不存在")
                    ids.append(found.id)
                cycle = self.board.find_cycle(task.id, ids)
                if cycle is not None:
                    raise BoardError("会形成循环依赖：" + " → ".join(cycle))
                fields["depends_on"] = ids
            if note:
                fields["inbox"] = [*task.inbox, note]
            if not fields:
                return f"{task.id} 没有要改的内容"
            self.board.update(task.id, **fields)
            return f"{task.id} 已更新"
        if task.terminal:
            return f"{task.id} 已结束，不能修改；用 retry"
        text = "；".join(x for x in [brief if isinstance(brief, str) else "", note] if x)
        return self.send(task.id, f"任务说明更新：{text}", mode="steer")

    def _approve(self, task: Task) -> str:
        if task.status != "proposed":
            return f"{task.id} 不是待确认任务"
        self.board.update(task.id, status="pending")
        return f"{task.id} 已批准"

    def _reject(self, task: Task, note: str) -> str:
        if task.status != "proposed":
            return f"{task.id} 不是待确认任务"
        self.board.update(
            task.id,
            status="cancelled",
            outcome=Outcome(status="cancelled", reason=note or "未被批准"),
        )
        return f"{task.id} 已拒绝"

    def _merge(self, task: Task) -> str:
        if not task.worktree:
            return f"{task.id} 不是 worktree 任务，改动已在工作区里"
        if task.status != "done":
            return f"{task.id} 还没完成，不能合并"
        branch = wt.branch_name(self.main.session_id, task.id)
        try:
            ok, output = wt.merge(self.workdir, branch)
        except wt.WorktreeError as e:
            return f"{task.id} 合并失败：{e}"
        if ok:
            wt.remove(self.workdir, Path(task.worktree))
            self.board.update(task.id, worktree=None)
            return f"{task.id} 已合并分支 {branch}"
        return f"{task.id} 合并有冲突，已中止：\n{output[:2000]}"

    def send(self, ref: str, text: str, *, mode: str = "steer", by: str = "main") -> str:
        task = self.board.find(ref)
        if task is None:
            return f"没有任务或子代理 {ref}"
        if not text.strip():
            return "消息为空"
        if task.terminal:
            return (
                f"{task.id} 已结束（{STATUS_LABEL.get(task.status)}）；"
                "需要它继续请用 task_control retry"
            )
        prefix = "（用户）" if by == "user" else "（主 agent）"
        runner = self._runners.get(task.id)
        if task.status == "running" and runner is not None and runner.agent is not None:
            runner.agent.steer(prefix + text, mode=FOLLOW_UP if mode == "follow_up" else STEER)
            result = f"已发给 {task.id}（{task.agent_name}）"
        else:
            self.board.update(task.id, inbox=[*task.inbox, prefix + text])
            result = f"{task.id} 当前{STATUS_LABEL.get(task.status)}，消息会在它继续时送达"
        if by == "user":
            self._notice(
                "",
                "info",
                f'<subagent_update kind="info" task="{task.id}">'
                f"用户直接对 {task.agent_name} 说：{text}</subagent_update>",
                headline=f"@{task.agent_name} ← {text[:80]}",
            )
        return result

    def answer(self, qid: str, text: str) -> str:
        task = self.board.answer(qid.strip(), text)
        return f"已回答 {task.id} 的问题，它会带着回答继续"

    # -- verbs: subagents --------------------------------------------------------

    def child_checkpoint(
        self, task_id: str, phase: str, summary: str, files: list[str], progress: float | None
    ) -> None:
        self.board.add_checkpoint(task_id, phase or "进行中", summary, files, progress)
        task = self.board.get(task_id)
        self._notice(task_id, "progress", "", headline=f"{task_id} {task.agent_name} 打点：{phase}")

    def child_need(
        self,
        task_id: str,
        *,
        on: list[str],
        new: dict[str, Any] | None,
        reason: str,
        policy: str,
        join: str,
    ) -> str:
        task = self.board.get(task_id)
        if policy == "cancel_self":
            self.board.update(
                task_id,
                status="cancelled",
                outcome=Outcome(status="cancelled", reason=f"子代理判断任务已无必要：{reason}"),
            )
            self._push(self._result_notice(self.board.get(task_id)))
            return "任务已取消。现在结束本轮。"
        if policy == "ask_main":
            detail = reason
            if on:
                detail += f"（涉及任务 {', '.join(on)}）"
            if new:
                detail += f"（需要新工作：{new.get('title', '')}）"
            return self.child_ask(task_id, detail)
        targets: list[str] = []
        for ref in on:
            found = self.board.find(ref)
            if found is None:
                raise BoardError(f"没有任务 {ref}；需要的工作还不存在时用 new=")
            targets.append(found.id)
        created: Task | None = None
        if new:
            if len(self.board.live()) >= self.settings.max_tasks:
                raise BoardError(f"任务数已到上限 {self.settings.max_tasks}，改用 ask_main")
            profile = resolve_profile(self.profiles, str(new.get("profile") or "worker"))
            status = "pending" if self.settings.auto == "on" else "proposed"
            scope = new.get("scope_paths") or []
            created = self.board.create(
                title=str(new.get("title") or reason[:60]),
                brief=str(new.get("brief") or reason),
                profile=profile.name,
                status=status,  # type: ignore[arg-type]
                created_by="subagent",
                scope_paths=[str(s) for s in scope] if isinstance(scope, list) else [],
                on_dep_fail=self.settings.on_dep_fail,
                max_tokens=self.settings.task_token_budget,
            )
            targets.append(created.id)
        if not targets:
            raise BoardError("need 需要 on=[任务] 或 new={...}")
        self.board.wait_on(
            task_id, targets, reason, "fail_fast" if join == "fail_fast" else "all_settled"
        )
        if created is not None and created.status == "proposed":
            q = (
                f"子代理 {task.agent_name}（{task_id}）需要一个新任务 "
                f"{created.id}「{created.title}」：{reason}。"
                f'用 task_control(id="{created.id}", action="approve"|"reject") 决定。'
            )
            self._push(
                Notice(
                    new_notice_id(f"{created.id}:approve"),
                    created.id,
                    "question",
                    f'<subagent_update task="{task_id}" kind="question">{q}</subagent_update>',
                    f"{task_id} 请求新任务 {created.id}：{created.title}",
                )
            )
        else:
            self._notice(
                task_id,
                "progress",
                "",
                headline=f"{task_id} 等待 {', '.join(targets)}：{reason[:80]}",
            )
        return (
            f"已挂起：等待 {', '.join(targets)}。它们结束后你会带着结果继续。"
            "现在结束本轮，不要再调用其他工具。"
        )

    def child_ask(self, task_id: str, question: str) -> str:
        task = self.board.get(task_id)
        q = self.board.ask(task_id, question)
        self._push(self._question_notice(task, q.id, question))
        return f"问题已提交（{q.id}）。回答后你会带着回答继续；现在结束本轮。"

    def child_finish(
        self,
        task_id: str,
        *,
        summary: str,
        files: list[str],
        follow_ups: list[str],
        failed: bool,
    ) -> str:
        self._finish(task_id, summary=summary, files=files, follow_ups=follow_ups, failed=failed)
        return "已交付。现在结束本轮。"

    def child_propose(self, task_id: str, title: str, brief: str, profile: str) -> str:
        if len(self.board.live()) >= self.settings.max_tasks:
            return f"任务数已到上限 {self.settings.max_tasks}，在 finish 的 follow_ups 里提出即可"
        created = self.board.create(
            title=title,
            brief=brief,
            profile=resolve_profile(self.profiles, profile).name,
            status="proposed",
            created_by="subagent",
            on_dep_fail=self.settings.on_dep_fail,
            max_tokens=self.settings.task_token_budget,
        )
        self._push(
            Notice(
                new_notice_id(f"{created.id}:approve"),
                created.id,
                "question",
                f'<subagent_update task="{task_id}" kind="question">'
                f"子代理建议新任务 {created.id}「{title}」：{brief[:400]}。"
                f'用 task_control(id="{created.id}", action="approve"|"reject") 决定。'
                "</subagent_update>",
                f"{task_id} 建议新任务 {created.id}：{title}",
            )
        )
        return f"已提议 {created.id}，由主 agent 决定；继续你自己的任务。"

    def _finish(
        self,
        task_id: str,
        *,
        summary: str,
        files: list[str] | None = None,
        follow_ups: list[str] | None = None,
        failed: bool = False,
        reason: str = "",
        auto: bool = False,
    ) -> None:
        task = self.board.get(task_id)
        if task.terminal:
            return
        status = "failed" if failed else "done"
        files = list(files or [])
        if task.worktree and not failed:
            try:
                stat = wt.commit_all(Path(task.worktree), f"{task.id}: {task.title}")
            except wt.WorktreeError as e:
                stat = f"（提交失败：{e}）"
            if stat:
                summary = f"{summary}\n\nworktree 改动：\n{stat}"
        outcome = Outcome(
            status=status,  # type: ignore[arg-type]
            summary=summary,
            files_changed=files,
            follow_ups=list(follow_ups or []),
            reason=reason or ("子代理结束时没有调用 finish，以最后的回复作为结果" if auto else ""),
        )
        owned_live = [t for t in self.board.owned_by(task_id) if not t.terminal]
        self.board.update(
            task_id,
            status="completing" if owned_live else status,
            outcome=outcome,
            waiting_on=[],
        )
        self._push(self._result_notice(self.board.get(task_id)))

    # -- read side -------------------------------------------------------------

    def status_text(self, ref: str | None = None) -> str:
        if ref:
            task = self.board.find(ref)
            if task is None:
                return f"没有任务或子代理 {ref}"
            return self._detail(task)
        if not self.board.tasks:
            return "任务板是空的。"
        lines = []
        for t in self.board.tasks.values():
            line = f"{t.id} [{STATUS_LABEL.get(t.status, t.status)}] {t.agent_name} · {t.title}"
            if t.depends_on and t.status in ("pending", "proposed"):
                line += f"（依赖 {', '.join(t.depends_on)}）"
            if t.status == "waiting":
                line += f"（{t.wait_reason or '等待 ' + ', '.join(t.waiting_on)}）"
            cp = t.last_checkpoint
            if cp is not None and not t.terminal:
                line += f" · 打点：{cp.phase}"
            if t.outcome is not None and t.terminal:
                head = (t.outcome.reason or t.outcome.summary).strip().splitlines()
                if head:
                    line += f" · {head[0][:100]}"
            lines.append(line)
        return "\n".join(lines)

    def _detail(self, task: Task) -> str:
        lines = [
            f"{task.id} {task.title}",
            f"状态：{STATUS_LABEL.get(task.status, task.status)}；角色：{task.profile}；"
            f"子代理：{task.agent_name}",
        ]
        if task.depends_on:
            lines.append(f"依赖：{', '.join(task.depends_on)}")
        if task.waiting_on:
            lines.append(f"等待：{', '.join(task.waiting_on)}（{task.wait_reason}）")
        if task.scope_paths:
            lines.append(f"认领路径：{', '.join(task.scope_paths)}")
        lines.append(f"说明：{task.brief[:1500]}")
        for cp in task.checkpoints[-5:]:
            lines.append(
                f"打点 {cp.seq}：{cp.phase}" + (f" — {cp.summary[:200]}" if cp.summary else "")
            )
        for q in task.questions:
            lines.append(
                f"问题 {q.id}：{q.text}"
                + (f" → {q.answer}" if q.answer is not None else "（未回答）")
            )
        if task.outcome is not None:
            lines.append(
                f"结果：{task.outcome.status} {task.outcome.reason}\n{task.outcome.summary[:2000]}"
            )
        if task.session_file:
            lines.append(f"记录：{task.session_file}")
        tokens = self.tokens(task.id)
        if tokens:
            lines.append(f"tokens：{tokens}")
        return "\n".join(lines)

    def summary_counts(self) -> dict[str, int]:
        counts = self.board.counts()
        return {k: v for k, v in counts.items() if k not in TERMINAL}

    def transcript(self, ref: str, limit: int = 40) -> list[tuple[str, str]]:
        """(role, text) pairs of a subagent's history for the UI viewer."""
        task = self.board.find(ref)
        if task is None or not task.session_file:
            return []
        runner = self._runners.get(task.id)
        if runner is not None and runner.agent is not None:
            messages = runner.agent.messages
        else:
            path = Path(task.session_file)
            if not path.exists():
                return []
            from limbo.sessions import load_session

            _, messages, _ = load_session(path)
        out: list[tuple[str, str]] = []
        for m in messages[1:]:
            if m.role == "tool":
                out.append(("tool", (m.content or "")[:300]))
            elif m.role == "assistant":
                if m.content:
                    out.append(("assistant", m.content))
                for tc in m.tool_calls or []:
                    out.append(("call", f"{tc['function']['name']}"))
            else:
                out.append((m.role, m.content or ""))
        return out[-limit:]


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p and p.strip())


def _result_id(task: Task) -> str:
    return f"{task.id}:result:{task.attempt}"


def _last_answer(agent: Agent) -> str:
    for m in reversed(agent.messages):
        if m.role == "assistant" and m.content:
            return m.content.strip()
    return ""


def _topo(specs: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    """Order plan entries so every in-plan dependency comes first."""
    by_key = {str(s["key"]): s for s in specs if s.get("key")}
    done: set[int] = set()
    visiting: set[int] = set()
    out: list[dict[str, Any]] = []

    def visit(spec: dict[str, Any], path: list[str]) -> str | None:
        ident = id(spec)
        if ident in done:
            return None
        if ident in visiting:
            return "计划里有循环依赖：" + " → ".join(path)
        visiting.add(ident)
        for dep in spec.get("depends_on") or []:
            target = by_key.get(str(dep))
            if target is not None:
                err = visit(target, [*path, str(dep)])
                if err:
                    return err
        visiting.discard(ident)
        done.add(ident)
        out.append(spec)
        return None

    for spec in specs:
        err = visit(spec, [str(spec.get("key") or spec.get("title"))])
        if err:
            return err
    return out
