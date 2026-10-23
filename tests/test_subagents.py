"""End-to-end tests for subagents: Conversation + Supervisor + real Agents.

A routing fake LLM answers each request by who is asking (the main agent
or the subagent of task tN, read from the system prompt) and by the last
message, so concurrently running agents can share one scripted client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from limbo.config import Config
from limbo.conversation import Conversation
from limbo.models import TextChunk, ToolCallEvent
from limbo.subagents.events import MainWake, PlanProposed


def _system(messages) -> str:
    return messages[0].content or "" if messages and messages[0].role == "system" else ""


def _who(messages) -> str:
    system = _system(messages)
    marker = "working on task "
    if "<subagent_role>" in system and marker in system:
        return system.split(marker, 1)[1].split(":", 1)[0]
    return "main"


class RoutingLLM:
    """``script[who]`` is a callable (messages, n) -> list of events."""

    def __init__(self, script, gates=None):
        self.script = script
        # who -> Event awaited before that agent's first request.
        self.gates = gates or {}
        self.calls: dict[str, int] = {}
        self.inputs: dict[str, list[str]] = {}

    async def chat(self, messages, tools, on_request=None):
        who = _who(messages)
        if who in self.gates and not self.calls.get(who):
            await self.gates[who].wait()
        n = self.calls.get(who, 0)
        self.calls[who] = n + 1
        last_user = next((m.content or "" for m in reversed(messages) if m.role == "user"), "")
        self.inputs.setdefault(who, []).append(last_user)
        await asyncio.sleep(0)
        for event in self.script[who](messages, n):
            yield event


def call(name, **arguments):
    return [
        ToolCallEvent(
            id=f"c-{name}-{json.dumps(arguments, sort_keys=True)[:20]}",
            name=name,
            arguments=arguments,
        )
    ]


def say(text):
    return [TextChunk(text=text)]


def make(tmp_path: Path, script, gates=None, **subagents) -> tuple[Conversation, RoutingLLM, list]:
    config = Config()
    for key, value in subagents.items():
        setattr(config.subagents, key, value)
    llm = RoutingLLM(script, gates)
    conv = Conversation(
        config=config,
        workdir=tmp_path,
        llm_client=llm,
        session_dir=tmp_path / "sessions",
    )
    events: list = []
    conv.subscribe(events.append)
    conv.start_background()
    return conv, llm, events


async def run_main(conv: Conversation, text: str) -> None:
    await conv.drive(conv.pump.run(text), lambda e: None)


async def settle(conv: Conversation) -> None:
    sup = conv.supervisor
    assert sup is not None
    await sup.wait_idle(timeout=10)


PLAN = {
    "tasks": [
        {
            "key": "a",
            "title": "Find the parser",
            "brief": "Locate the parser.",
            "profile": "explore",
        },
        {
            "key": "b",
            "title": "Explain the parser",
            "brief": "Explain it.",
            "profile": "explore",
            "depends_on": ["a"],
        },
    ]
}


def two_task_script():
    def main(messages, n):
        last = messages[-1]
        if last.role == "user" and "<subagent_updates" in (last.content or ""):
            return say("Summary: both done.")
        if last.role == "tool":
            return say("Delegated.")
        return call("plan_tasks", **PLAN)

    def t1(messages, n):
        if n == 0:
            return call("checkpoint", phase="searching", summary="grep for parse")
        if n == 1:
            return call("finish", summary="Parser lives in src/parse.py", files_changed=[])
        return say("unexpected")

    def t2(messages, n):
        if n == 0:
            return call("finish", summary="It is a recursive-descent parser")
        return say("unexpected")

    return {"main": main, "t1": t1, "t2": t2}


@pytest.mark.asyncio
async def test_plan_runs_dag_and_reports_back(tmp_path):
    conv, llm, events = make(tmp_path, two_task_script(), auto="on")
    await run_main(conv, "go")
    await settle(conv)
    sup = conv.supervisor
    assert sup is not None
    board = sup.board
    assert [t.status for t in board.tasks.values()] == ["done", "done"]
    # t2 waited for t1 and started with its result.
    assert "Parser lives in src/parse.py" in llm.inputs["t2"][0]
    assert board.tasks["t1"].checkpoints[0].phase == "searching"
    # Wave done: the idle main agent is woken with both results.
    assert any(isinstance(e, MainWake) for e in events)
    wake = conv.take_wake_input()
    assert wake is not None and "<subagent_updates" in wake
    assert "Parser lives" in wake and "recursive-descent" in wake
    await run_main(conv, wake)
    assert conv.agent.messages[-1].content == "Summary: both done."
    # Subagent transcripts are their own session files.
    assert Path(board.tasks["t1"].session_file).exists()
    conv.close()

    # Reload: the board comes back from the session file, nothing is
    # delivered twice.
    path = conv.agent.session_path
    conv2 = Conversation(
        config=conv.config,
        workdir=tmp_path,
        llm_client=llm,
        session_dir=tmp_path / "sessions",
        resume=path,
    )
    sup2 = conv2.supervisor
    assert sup2 is not None
    assert [t.status for t in sup2.board.tasks.values()] == ["done", "done"]
    assert sup2.pending_notices() == []
    conv2.close()


@pytest.mark.asyncio
async def test_suggest_mode_waits_for_approval(tmp_path):
    conv, llm, events = make(tmp_path, two_task_script(), auto="suggest")
    await run_main(conv, "go")
    sup = conv.supervisor
    assert sup is not None
    # plan_tasks ended the main turn (no "Delegated." follow-up call).
    assert llm.calls["main"] == 1
    proposed = [e for e in events if isinstance(e, PlanProposed)]
    assert proposed and proposed[0].task_ids == ["t1", "t2"]
    await asyncio.sleep(0.05)
    assert "t1" not in llm.calls
    assert sup.control("all", "approve", by="user").count("已批准") == 2
    await settle(conv)
    assert [t.status for t in sup.board.tasks.values()] == ["done", "done"]
    conv.close()


@pytest.mark.asyncio
async def test_need_suspends_until_dependency_finishes(tmp_path):
    release = asyncio.Event()

    def main(messages, n):
        if messages[-1].role == "tool":
            return say("ok")
        return call(
            "plan_tasks",
            tasks=[
                {"key": "ui", "title": "UI", "brief": "Build the button.", "profile": "worker"},
                {"key": "api", "title": "API", "brief": "Build the endpoint.", "profile": "worker"},
            ],
        )

    def t1(messages, n):
        if n == 0:
            return call("need", on=["t2"], reason="needs the response format")
        return call("finish", summary="button done using " + (messages[-1].content or "")[:200])

    def t2(messages, n):
        if n == 0:
            return call("checkpoint", phase="writing endpoint")
        return call("finish", summary="endpoint returns CSV")

    conv, llm, _ = make(tmp_path, {"main": main, "t1": t1, "t2": t2}, auto="on")
    await run_main(conv, "go")
    await settle(conv)
    board = conv.supervisor.board
    t1 = board.tasks["t1"]
    assert t1.status == "done"
    # Resumed with the dependency's result.
    assert "endpoint returns CSV" in llm.inputs["t1"][-1]
    assert "endpoint returns CSV" in t1.outcome.summary
    release.set()
    conv.close()


@pytest.mark.asyncio
async def test_cycle_is_refused(tmp_path):
    def main(messages, n):
        if messages[-1].role == "tool":
            return say("ok")
        return call(
            "plan_tasks",
            tasks=[
                {"key": "a", "title": "A", "brief": "a"},
                {"key": "b", "title": "B", "brief": "b", "depends_on": ["a"]},
            ],
        )

    def t1(messages, n):
        if n == 0:
            return call("need", on=["t2"], reason="loop")
        if n == 1:
            # The refusal comes back as a tool error; give up cleanly.
            assert "循环依赖" in (messages[-1].content or "")
            return call("finish", summary="gave up", status="failed")
        return say("x")

    def t2(messages, n):
        return call("finish", summary="b done")

    conv, _, _ = make(tmp_path, {"main": main, "t1": t1, "t2": t2}, auto="on")
    await run_main(conv, "go")
    await settle(conv)
    board = conv.supervisor.board
    assert board.tasks["t1"].status == "failed"
    # t2 depended on t1: notified (default), so it still ran.
    assert board.tasks["t2"].status == "done"
    conv.close()


@pytest.mark.asyncio
async def test_ask_main_and_answer(tmp_path):
    def main(messages, n):
        last = messages[-1]
        if last.role == "user" and 'kind="question"' in (last.content or ""):
            qid = (last.content or "").split('id="', 1)[1].split('"', 1)[0]
            return call("answer", question_id=qid, text="Use CSV")
        if last.role == "tool" and "plan" not in (last.content or "") and n > 1:
            return say("answered")
        if last.role == "tool":
            return say("delegated")
        return call("plan_tasks", tasks=[{"title": "Export", "brief": "Export data"}])

    def t1(messages, n):
        if n == 0:
            return call("ask_main", question="CSV or XLSX?")
        return call("finish", summary="exported as " + (messages[-1].content or "")[-40:])

    gate = asyncio.Event()
    conv, llm, events = make(tmp_path, {"main": main, "t1": t1}, {"t1": gate}, auto="on")
    await run_main(conv, "go")
    gate.set()  # t1 asks only once the main agent is idle
    await asyncio.sleep(0.1)
    assert conv.supervisor.board.tasks["t1"].status == "waiting"
    # A question wakes the idle main agent at once.
    assert any(isinstance(e, MainWake) for e in events)
    wake = conv.take_wake_input()
    assert wake and "CSV or XLSX?" in wake
    await run_main(conv, wake)
    await settle(conv)
    t1 = conv.supervisor.board.tasks["t1"]
    assert t1.status == "done"
    assert "Use CSV" in llm.inputs["t1"][-1]
    conv.close()


@pytest.mark.asyncio
async def test_user_pause_resume_and_send(tmp_path):
    gate = asyncio.Event()

    def main(messages, n):
        if messages[-1].role == "tool":
            return say("ok")
        return call("plan_tasks", tasks=[{"title": "Long job", "brief": "work"}])

    state = {"n": 0}

    def t1(messages, n):
        state["n"] += 1
        if n == 0:
            return call("checkpoint", phase="phase 1")
        return call("finish", summary="done after: " + (messages[-1].content or "")[:300])

    conv, llm, _ = make(tmp_path, {"main": main, "t1": t1}, auto="on", max_parallel=1)
    sup = conv.supervisor
    await run_main(conv, "go")
    # Pause before it gets far, then talk to it and resume.
    out = sup.control("t1", "pause", note="wait for me", by="user")
    assert "已暂停" in out
    await asyncio.sleep(0.05)
    t1 = sup.board.tasks["t1"]
    assert t1.status == "paused"
    assert "消息会在它继续时送达" in sup.send("@worker-1", "use the new API", by="user")
    sup.control("t1", "resume", by="user")
    await settle(conv)
    t1 = sup.board.tasks["t1"]
    assert t1.status == "done"
    assert "use the new API" in llm.inputs["t1"][-1]
    # The main agent is told about the user's direct actions (info).
    conv.supervisor.flush_notices()
    queued = [i.text for i in conv.agent.queued_items]
    assert any("用户直接" in q for q in queued)
    gate.set()
    conv.close()


@pytest.mark.asyncio
async def test_restart_pauses_interrupted_tasks(tmp_path):
    block = asyncio.Event()

    class SlowLLM(RoutingLLM):
        async def chat(self, messages, tools, on_request=None):
            if _who(messages) == "t1" and self.calls.get("t1", 0) == 0:
                self.calls["t1"] = 1
                await block.wait()  # never released: the process "dies"
            async for e in super().chat(messages, tools, on_request):
                yield e

    def main(messages, n):
        if messages[-1].role == "tool":
            return say("ok")
        return call("plan_tasks", tasks=[{"title": "Job", "brief": "work"}])

    def t1(messages, n):
        return call("finish", summary="finished after restart")

    config = Config()
    config.subagents.auto = "on"
    llm = SlowLLM({"main": main, "t1": t1})
    conv = Conversation(
        config=config, workdir=tmp_path, llm_client=llm, session_dir=tmp_path / "sessions"
    )
    conv.start_background()
    await run_main(conv, "go")
    await asyncio.sleep(0.05)
    assert conv.supervisor.board.tasks["t1"].status == "running"
    path = conv.agent.session_path
    conv.close()  # stop(): no outcome written
    await asyncio.sleep(0.05)

    conv2 = Conversation(
        config=config,
        workdir=tmp_path,
        llm_client=llm,
        session_dir=tmp_path / "sessions",
        resume=path,
    )
    conv2.start_background()
    sup = conv2.supervisor
    assert sup.board.tasks["t1"].status == "paused"
    assert sup.interrupted_on_load == ["t1"]
    sup.control("all", "resume", by="user")
    await settle(conv2)
    assert sup.board.tasks["t1"].status == "done"
    conv2.close()


@pytest.mark.asyncio
async def test_subagents_off_installs_no_tools(tmp_path):
    config = Config()
    config.subagents.auto = "off"
    conv = Conversation(
        config=config,
        workdir=tmp_path,
        llm_client=RoutingLLM({}),
        session_dir=tmp_path / "sessions",
    )
    assert conv.agent.registry.get("plan_tasks") is None
    config2 = Config()
    conv2 = Conversation(
        config=config2,
        workdir=tmp_path,
        llm_client=RoutingLLM({}),
        session_dir=tmp_path / "sessions",
    )
    assert conv2.agent.registry.get("plan_tasks") is not None
    assert "<available_agents>" in (conv2.agent.messages[0].content or "")
    conv.close()
    conv2.close()


# -- frontends -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_headless_waits_for_subagents_and_runs_the_wake_turn(tmp_path):
    from limbo.headless import Eof, HeadlessFrontend, Submit

    config = Config()
    config.subagents.auto = "on"
    llm = RoutingLLM(two_task_script())
    conv = Conversation(
        config=config, workdir=tmp_path, llm_client=llm, session_dir=tmp_path / "sessions"
    )
    chunks: list[str] = []

    class Reporter:
        def report(self, *a, **k): ...
        def report_session(self, *a, **k): ...
        def release(self): ...

    frontend = HeadlessFrontend(
        conversation=conv, reporters=Reporter(), out=chunks.append, echo_input=True
    )

    async def inputs():
        yield Submit("go")
        yield Submit("/task")
        yield Eof()

    rc = await asyncio.wait_for(frontend.run(inputs()), 10)
    out = "".join(chunks)
    assert rc == 0
    assert "✓ t1 explore-1 完成" in out
    assert "子代理有新进展" in out
    # EOF waited for the subagents and the main agent's summary turn.
    assert "Summary: both done." in out


@pytest.mark.asyncio
async def test_tui_direct_message_panel_and_wake(tmp_path):
    from limbo.ui.app import LimboApp
    from limbo.ui.screens.agents_panel import AgentsPanel
    from limbo.ui.widgets.pills import Pills

    gate = asyncio.Event()

    def main(messages, n):
        last = messages[-1]
        if last.role == "user" and "<subagent_updates" in (last.content or ""):
            return say("main saw the result")
        if last.role == "tool":
            return say("delegated")
        return call("plan_tasks", tasks=[{"title": "Job", "brief": "work", "profile": "worker"}])

    def t1(messages, n):
        return call("finish", summary="job finished: " + (messages[-1].content or "")[-60:])

    llm = RoutingLLM({"main": main, "t1": t1}, {"t1": gate})
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.subagents.auto = "on"
    app = LimboApp(workdir=tmp_path, config=cfg, llm_client=llm, session_dir=tmp_path / "s")
    async with app.run_test() as pilot:
        screen = app.screen_stack[-1]
        screen.on_user_submitted(
            type(
                "E",
                (),
                {
                    "message": "go",
                    "attachments": [],
                    "force_text": False,
                    "followup": False,
                },
            )()
        )

        async def until(cond):
            for _ in range(300):
                await pilot.pause()
                if cond():
                    return
            raise AssertionError("timeout")

        sup = screen.conversation.supervisor
        await until(lambda: "t1" in sup.board.tasks and not screen.conversation.busy)
        assert "子代理" in screen.query_one("#pills", Pills).plain
        # @name reaches the running subagent (its steer queue).
        screen.on_user_submitted(
            type(
                "E",
                (),
                {
                    "message": "@worker-1 use tabs",
                    "attachments": [],
                    "force_text": False,
                    "followup": False,
                },
            )()
        )
        child = sup._runners["t1"].agent
        assert any("use tabs" in item.text for item in child.queued_items)
        screen._agents_cmd("")
        await pilot.pause()
        assert isinstance(app.screen, AgentsPanel)
        await pilot.press("escape")
        gate.set()
        await until(lambda: sup.board.tasks["t1"].status == "done")
        await until(lambda: "main saw the result" in str(screen.agent.messages[-1].content))


@pytest.mark.asyncio
async def test_need_new_work_and_cancel_self(tmp_path):
    def main(messages, n):
        if messages[-1].role == "tool":
            return say("ok")
        return call(
            "plan_tasks",
            tasks=[
                {"title": "Feature", "brief": "build it"},
                {"title": "Obsolete", "brief": "maybe not needed"},
            ],
        )

    def t1(messages, n):
        if n == 0:
            return call("need", new={"title": "Schema", "brief": "add the table"}, reason="no db")
        return call("finish", summary="feature on top of: " + (messages[-1].content or "")[:300])

    def t2(messages, n):
        return call("need", policy="cancel_self", reason="t1 covers it")

    def t3(messages, n):
        return call("finish", summary="table added")

    conv, llm, _ = make(tmp_path, {"main": main, "t1": t1, "t2": t2, "t3": t3}, auto="on")
    await run_main(conv, "go")
    await settle(conv)
    board = conv.supervisor.board
    assert board.tasks["t3"].created_by == "subagent"
    assert board.tasks["t1"].status == "done"
    assert "table added" in board.tasks["t1"].outcome.summary
    assert board.tasks["t2"].status == "cancelled"
    conv.close()


# -- token budget ----------------------------------------------------------------


def _usage(total, cached):
    from limbo.models import CompletionMeta

    return CompletionMeta(
        usage={
            "prompt_tokens": total - 10,
            "completion_tokens": 10,
            "total_tokens": total,
            "prompt_cache_hit_tokens": cached,
        }
    )


def _budget_script(child):
    one_task = {"tasks": [{"title": "Work", "brief": "Do it.", "profile": "worker"}]}

    def main(messages, n):
        last = messages[-1]
        if last.role == "user" and "<subagent_updates" in (last.content or ""):
            return say("ok")
        if last.role == "tool":
            return say("Delegated.")
        return call("plan_tasks", **one_task)

    return {"main": main, "t1": child}


@pytest.mark.asyncio
async def test_budget_ignores_prompt_cache_hits(tmp_path):
    # Every request re-sends a big cached prefix: 3 x 900 = 2700 total
    # tokens, but only 3 x 50 fresh ones — well inside a 1000 budget.
    def t1(messages, n):
        if n < 2:
            return [*call("checkpoint", phase=f"p{n}", summary="s"), _usage(900, 850)]
        return [*call("finish", summary="all done"), _usage(900, 850)]

    conv, llm, _ = make(tmp_path, _budget_script(t1), auto="on", task_token_budget=1000)
    await run_main(conv, "go")
    await settle(conv)
    task = conv.supervisor.board.tasks["t1"]  # type: ignore[union-attr]
    assert task.status == "done" and task.outcome.summary == "all done"  # type: ignore[union-attr]
    assert llm.calls["t1"] == 3
    conv.close()


@pytest.mark.asyncio
async def test_over_budget_finishes_the_batch_then_stops(tmp_path):
    # The second response crosses the budget: its write still runs (no
    # "not executed" placeholder), no third request goes out, and the
    # outcome names the budget and the change made.
    def t1(messages, n):
        if n == 0:
            return [*call("write", path="a.txt", content="a"), _usage(600, 0)]
        if n == 1:
            return [*call("write", path="b.txt", content="b"), _usage(600, 0)]
        return say("unexpected")

    conv, llm, _ = make(tmp_path, _budget_script(t1), auto="on", task_token_budget=1000)
    await run_main(conv, "go")
    await settle(conv)
    task = conv.supervisor.board.tasks["t1"]  # type: ignore[union-attr]
    assert llm.calls["t1"] == 2
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    assert task.status == "done"
    reason = task.outcome.reason  # type: ignore[union-attr]
    assert "超出 token 预算（新增 1200 > 1000" in reason
    assert "write：" in reason
    child_history = Path(task.session_file).read_text(encoding="utf-8")
    assert "未执行" not in child_history
    conv.close()
