"""Unit tests for the task board, reconcile and the subagent helpers."""

from __future__ import annotations

import pytest

from limbo.subagents.board import (
    BoardError,
    CancelTask,
    Deadlock,
    MakeReady,
    Start,
    TaskBoard,
    reconcile,
)
from limbo.subagents.hooks import ClaimHook, NoEditHook, matches
from limbo.subagents.profiles import discover_profiles, parse_profile_md, resolve_profile
from limbo.subagents.project_scan import format_facts, scan_project, suggest_profiles
from limbo.tools.hooks import Block


def board_with(*titles: str) -> TaskBoard:
    board = TaskBoard()
    for title in titles:
        board.create(title=title)
    return board


def test_records_replay_to_the_same_board():
    records: list[dict] = []
    board = TaskBoard(sink=records.append)
    board.create(title="A")
    board.create(title="B", depends_on=["t1"])
    board.add_checkpoint("t1", "phase", "summary")
    q = board.ask("t2", "why?")
    board.answer(q.id, "because")
    replay = TaskBoard()
    for rec in records:
        replay.apply(rec)
    assert replay.snapshot() == board.snapshot()
    # A snapshot record folds everything before it.
    folded = TaskBoard()
    folded.apply({"type": "task", "op": "snapshot", **board.snapshot()})
    assert folded.snapshot() == board.snapshot()
    assert folded.next_id() == "t3"


def test_wait_on_refuses_cycles():
    board = board_with("A")
    board.create(title="B", depends_on=["t1"])
    with pytest.raises(BoardError, match="循环依赖"):
        board.wait_on("t1", ["t2"])
    board.create(title="C")
    board.wait_on("t1", ["t3"])
    assert board.get("t1").status == "waiting"


def test_reconcile_orders_ready_then_start_within_slots():
    board = board_with("A", "B", "C")
    board.create(title="D", depends_on=["t1"])
    actions = reconcile(board, max_parallel=2)
    assert {a.task_id for a in actions if isinstance(a, MakeReady)} == {"t1", "t2", "t3"}
    for a in actions:
        board.set_status(a.task_id, "ready")
    starts = reconcile(board, max_parallel=2)
    # t1 is on the longer critical path (t4 depends on it): first.
    assert starts == [Start("t1"), Start("t2")]


def test_owner_cascade_and_dependency_policies():
    board = board_with("parent")
    board.create(title="child", owner="t1", status="running")
    board.set_status("t1", "cancelled")
    assert reconcile(board, max_parallel=3) == [CancelTask("t2", "所属任务 t1 已cancelled")]

    board = board_with("dep")
    board.create(title="notify", depends_on=["t1"])
    board.create(title="cascade", depends_on=["t1"], on_dep_fail="cascade")
    board.set_status("t1", "failed")
    actions = reconcile(board, max_parallel=3)
    assert MakeReady in {type(a) for a in actions}
    assert CancelTask("t3", "前置依赖失败或被取消") in actions


def test_fail_fast_join_wakes_on_first_failure():
    board = board_with("waiter", "a", "b")
    board.wait_on("t1", ["t2", "t3"], join="fail_fast")
    board.set_status("t2", "failed")
    board.set_status("t3", "running")
    actions = reconcile(board, max_parallel=3)
    assert [type(a) for a in actions] == [MakeReady]
    board = board_with("waiter", "a", "b")
    board.wait_on("t1", ["t2", "t3"])  # all_settled
    board.set_status("t2", "failed")
    board.set_status("t3", "running")
    assert not any(isinstance(a, MakeReady) for a in reconcile(board, max_parallel=3))


def test_deadlock_is_reported_but_paused_work_is_not_stuck():
    board = board_with("a", "b")
    board.update("t1", status="waiting", waiting_on=["t2"])
    board.update("t2", status="waiting", waiting_on=["t1"])
    actions = reconcile(board, max_parallel=3)
    assert len(actions) == 1 and isinstance(actions[0], Deadlock)
    board.create(title="c", status="paused")
    assert reconcile(board, max_parallel=3) == []


def test_claim_hook_blocks_other_tasks_paths(tmp_path):
    board = TaskBoard()
    board.create(title="api", scope_paths=["src/api/**"], status="running")
    board.create(title="ui", status="running")
    hook = ClaimHook("t2", board, tmp_path)
    blocked = hook.before_tool("edit", {"path": "src/api/routes.py"})
    assert isinstance(blocked, Block) and "t1" in blocked.reason
    assert hook.before_tool("edit", {"path": "src/ui/app.py"}) is None
    assert hook.before_tool("read", {"path": "src/api/routes.py"}) is None
    assert ClaimHook("t1", board, tmp_path).before_tool("edit", {"path": "src/api/x"}) is None
    assert matches("src/api", ["src/api/**"])
    assert isinstance(NoEditHook().before_tool("write", {}), Block)


def test_profiles_discovery_and_overrides(tmp_path):
    agents = tmp_path / ".agents" / "agents"
    agents.mkdir(parents=True)
    (agents / "explore.md").write_text(
        "---\nname: explore\ndescription: mine\ntools: read_only\nmodel: m-1\n---\nBody\n"
    )
    profiles = discover_profiles(tmp_path, user_dir=tmp_path / "none")
    assert profiles["explore"].description == "mine"
    assert profiles["explore"].model == "m-1"
    assert "verifier" in profiles
    assert resolve_profile(profiles, "unknown").name == "unknown"
    p = parse_profile_md("---\ntools: read,grep\nscope_paths: [a/**, b]\n---\nx", "n")
    assert p.tool_names == ["read", "grep"] and p.scope_paths == ("a/**", "b")
    assert not p.writes


def test_project_scan_suggests_profiles(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"workspaces": ["packages/*"], "scripts": {"test": "vitest"}}'
    )
    for name in ("web", "api"):
        pkg = tmp_path / "packages" / name
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text("{}")
        (pkg / "index.ts").write_text("")
    facts = scan_project(tmp_path)
    assert facts.packages == ["packages/api", "packages/web"]
    assert facts.check_commands == ["npm run test"]
    assert "TypeScript" in format_facts(facts)
    names = [s.name for s in suggest_profiles(facts, tmp_path)]
    assert names == ["packages-api-worker", "packages-web-worker", "project-verifier"]


def test_worktree_create_commit_merge(tmp_path):
    import subprocess

    from limbo.subagents import worktree as wt

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (tmp_path / "a.txt").write_text("a\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    tree = wt.create(tmp_path, "session-123456789", "t1")
    (tree.path / "b.txt").write_text("b\n")
    assert "b.txt" in wt.commit_all(tree.path, "t1 work")
    ok, _ = wt.merge(tmp_path, tree.branch)
    assert ok and (tmp_path / "b.txt").exists()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert ".limbo" not in status
    wt.remove(tmp_path, tree.path)
