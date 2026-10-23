"""Daily mode P1: bundled skills, skill_manage, session_search, ask_user,
web_fetch PDF/long pages, memory nudge, /learn."""

from __future__ import annotations

import io
import json

import httpx
import pytest
import respx

import limbo.skills as skills_module
from limbo.agent import Agent
from limbo.config import Config
from limbo.conversation import Conversation
from limbo.memory import MEMORY_NUDGE
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent
from limbo.skills import discover_skills
from limbo.subagents.profiles import BUILTIN_PROFILES
from limbo.tools.registry import ToolRegistry
from limbo.tools.session_search import SessionSearchTool
from limbo.tools.skill_manage import SkillManageTool
from limbo.tools.web_fetch import WebFetchTool


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = 0
        self.seen: list[list] = []
        self.tools: list = []

    async def chat(self, messages, tools, on_request=None):
        self.calls += 1
        self.seen.append(list(messages))
        self.tools.append(tools)
        for event in self.responses.pop(0):
            yield event


def _skill_md(name, description="Use when testing.", body="- rule one\n"):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"


def _manage(tmp_path):
    return SkillManageTool(tmp_path, lambda: skills_module.DEFAULT_DAILY_SKILLS_DIR)


# -- bundled starter pack ---------------------------------------------------------


def test_bundled_skills_only_in_daily_and_replaceable(tmp_path):
    daily = {s.name: s for s in discover_skills(tmp_path, daily=True)}
    assert {"pdf", "docx", "xlsx", "grounded-citations", "arxiv"} <= set(daily)
    assert discover_skills(tmp_path) == []
    # Every bundled SKILL.md parses and carries its license note.
    for skill in daily.values():
        assert skill.description and (skill.path.parent / "LICENSE").is_file()
    own = skills_module.DEFAULT_DAILY_SKILLS_DIR / "pdf"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text(_skill_md("pdf", "My own PDF flow."))
    mine = {s.name: s for s in discover_skills(tmp_path, daily=True)}["pdf"]
    assert mine.description == "My own PDF flow."


# -- skill_manage --------------------------------------------------------------------


def test_skill_manage_create_then_patch(tmp_path):
    tool = _manage(tmp_path)
    r = tool.execute({"operations": [{"action": "create", "name": "weekly-report",
                                      "content": _skill_md("weekly-report")}]})
    assert r.success, r.error
    r = tool.execute({"operations": [
        {"action": "patch", "name": "weekly-report", "old_string": "- rule one",
         "new_string": "- rule one\n- keep it under 200 words"},
        {"action": "write_file", "name": "weekly-report",
         "file_path": "references/style.md", "content": "short"},
    ]})
    assert r.success, r.error
    root = skills_module.DEFAULT_DAILY_SKILLS_DIR / "weekly-report"
    assert "200 words" in (root / "SKILL.md").read_text()
    assert (root / "references" / "style.md").read_text() == "short"
    assert (root / "SKILL.md.bak").exists()


def test_skill_manage_patching_a_bundled_skill_copies_it_first(tmp_path):
    tool = _manage(tmp_path)
    bundled = skills_module.BUNDLED_DAILY_SKILLS_DIR / "arxiv" / "SKILL.md"
    before = bundled.read_text()
    line = next(ln for ln in before.splitlines() if ln.startswith("#"))
    r = tool.execute({"operations": [{"action": "patch", "name": "arxiv",
                                      "old_string": line, "new_string": line + " (mine)"}]})
    assert r.success, r.error
    assert bundled.read_text() == before  # the package is never edited
    copy = skills_module.DEFAULT_DAILY_SKILLS_DIR / "arxiv"
    assert "(mine)" in (copy / "SKILL.md").read_text()
    assert (copy / "scripts").is_dir()  # support files came along


def test_skill_manage_is_all_or_nothing_and_validates(tmp_path):
    tool = _manage(tmp_path)
    r = tool.execute({"operations": [
        {"action": "create", "name": "good-skill", "content": _skill_md("good-skill")},
        {"action": "write_file", "name": "good-skill", "file_path": "../escape.md",
         "content": "x"},
    ]})
    assert not r.success and "operation 2" in (r.error or "")
    assert not (skills_module.DEFAULT_DAILY_SKILLS_DIR / "good-skill").exists()
    r = tool.execute({"operations": [{"action": "create", "name": "a-skill",
                                      "content": _skill_md("other-name")}]})
    assert not r.success and "does not match" in (r.error or "")


# -- session_search ------------------------------------------------------------------


def _session(directory, sid, messages, daily=True):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "meta", "id": sid, "title": f"t-{sid}",
                         "created_at": "2026-10-01T00:00:00", "daily_mode": daily})]
    lines += [json.dumps({"role": r, "content": c}) for r, c in messages]
    (directory / f"{sid}.jsonl").write_text("\n".join(lines) + "\n")


def test_session_search_finds_and_reads(tmp_path):
    sessions = tmp_path / "sessions"
    _session(sessions, "s1", [("user", "plan a trip to Kyoto in spring"),
                              ("assistant", "Kyoto itinerary: day 1 temples")])
    _session(sessions, "s2", [("user", "fix the Kyoto parser bug")], daily=False)
    tool = SessionSearchTool(tmp_path, lambda: sessions)
    out = tool.execute({"query": "kyoto"}).output or ""
    assert "session s1" in out and "s2" not in out
    out = tool.execute({"query": "kyoto", "include_coding": True}).output or ""
    assert "session s2" in out
    read = tool.execute({"session_id": "s1", "around": 1}).output or ""
    assert "day 1 temples" in read
    assert "No past sessions" in (tool.execute({"query": "paris"}).output or "")


# -- ask_user ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_ends_the_turn(tmp_path):
    client = FakeLLMClient([[
        ToolCallEvent(id="c1", name="ask_user",
                      arguments={"question": "Which city?", "options": ["Kyoto", "Osaka"]}),
        CompletionMeta(),
    ]])
    agent = Agent(config=Config(), llm_client=client, workdir=tmp_path,
                  session_dir=tmp_path / "sessions", meta_fields={"daily_mode": True})
    _ = [e async for e in agent.run("plan my trip")]
    assert client.calls == 1 and agent.ended_by_tool
    assert "1. Kyoto" in (agent.messages[-1].content or "")


# -- web_fetch: PDF and long pages ----------------------------------------------------------


@respx.mock
def test_web_fetch_reads_pdf_and_saves_long_pages(tmp_path):
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    respx.get("https://x.example/a.pdf").mock(return_value=httpx.Response(
        200, content=buf.getvalue(), headers={"content-type": "application/pdf"}))
    respx.get("https://x.example/long").mock(return_value=httpx.Response(
        200, text="y" * 500, headers={"content-type": "text/plain"}))
    tool = WebFetchTool(tmp_path, max_chars=100)
    pdf = tool.execute({"url": "https://x.example/a.pdf"})
    assert pdf.success and "## Page 1" in (pdf.output or "")
    long = tool.execute({"url": "https://x.example/long"}).output or ""
    saved = list((tmp_path / "fetched").glob("*.md"))
    assert len(saved) == 1 and str(saved[0]) in long
    assert saved[0].read_text().endswith("y" * 500)


# -- memory nudge ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_nudge_after_quiet_turns(tmp_path):
    config = Config()
    config.daily.memory_nudge_turns = 1
    client = FakeLLMClient([[TextChunk(text="ok"), CompletionMeta()] for _ in range(3)])
    agent = Agent(config=config, llm_client=client, workdir=tmp_path,
                  session_dir=tmp_path / "sessions", meta_fields={"daily_mode": True})
    for text in ("one", "two"):
        _ = [e async for e in agent.run(text)]
    users = [m.content or "" for m in agent.messages if m.role == "user"]
    assert MEMORY_NUDGE not in users[0] and MEMORY_NUDGE in users[1]
    agent.registry.memory_written = True  # a memory write resets the count
    _ = [e async for e in agent.run("three")]
    assert MEMORY_NUDGE not in (agent.messages[-2].content or "")


# -- /learn and profiles -------------------------------------------------------------


def test_learn_plans_a_skill_review_in_daily_mode_only(tmp_path):
    conv = Conversation(config=Config(), workdir=tmp_path, llm_client=FakeLLMClient(),
                        session_dir=tmp_path / "sessions")
    assert "daily" in conv.learn()
    conv.agent.set_daily_mode(True)
    assert "复盘" in conv.learn()
    assert conv.supervisor is not None
    live = conv.supervisor.board.live()
    assert [t.profile for t in live] == ["skill-reviewer"]
    assert live[0].status == "pending" and live[0].context == "fork"
    assert "进行中" in conv.learn()
    conv.close()


def test_builtin_daily_profiles():
    profiles = {p.name: p for p in BUILTIN_PROFILES}
    assert profiles["researcher"].tools == "read_only"
    assert profiles["skill-reviewer"].tool_names == ["read", "ls", "find", "grep", "skill_manage"]


def test_daily_registry_has_all_p1_tools(tmp_path):
    registry = ToolRegistry(tmp_path, Config())
    registry.daily_mode = True
    for name in ("web_fetch", "memory", "session_search", "ask_user", "skill_manage"):
        assert registry.get(name) is not None
    registry.daily_mode = False
    assert registry.get("skill_manage") is None


def test_daily_children_get_web_fetch_but_not_main_only_tools(tmp_path):
    conv = Conversation(config=Config(), workdir=tmp_path, llm_client=FakeLLMClient(),
                        session_dir=tmp_path / "sessions", daily=True)
    assert conv.supervisor is not None
    conv.supervisor.plan([{"title": "compare", "brief": "b", "profile": "researcher"}],
                         created_by="user")
    task = conv.supervisor.board.live()[0]
    child, _ = conv.supervisor._open_agent(task)
    assert child.daily_mode and child.registry.get("web_fetch") is not None
    for name in ("memory", "ask_user", "skill_manage", "session_search"):
        assert child.registry.get(name) is None
    child.close()
    conv.close()


@pytest.mark.asyncio
async def test_learn_runs_a_reviewer_that_writes_a_skill(tmp_path):
    import asyncio

    client = FakeLLMClient([
        [ToolCallEvent(id="r1", name="skill_manage", arguments={"operations": [
            {"action": "create", "name": "trip-planning",
             "content": _skill_md("trip-planning", "Use when planning trips.")}]}),
         CompletionMeta()],
        [TextChunk(text="trip-planning: new"), CompletionMeta()],
    ])
    conv = Conversation(config=Config(), workdir=tmp_path, llm_client=client,
                        session_dir=tmp_path / "sessions", daily=True)
    conv.start_background()
    assert "复盘" in conv.learn()
    assert conv.supervisor is not None
    for _ in range(200):
        if not conv.supervisor.board.live():
            break
        await asyncio.sleep(0.01)
    task = next(iter(conv.supervisor.board.tasks.values()))
    assert task.status == "done", task
    assert task.outcome is not None and task.outcome.summary == "trip-planning: new"
    assert (skills_module.DEFAULT_DAILY_SKILLS_DIR / "trip-planning" / "SKILL.md").exists()
    # Cache parity: the reviewer sends the main session's exact system
    # prompt and tool list; its role and brief ride in the first message.
    assert client.seen[0][0].content == conv.agent.system_prompt
    assert client.tools[0] == conv.agent.tool_definitions()
    first = client.seen[0][-1].content or ""
    assert "<subagent_role>" in first and "skill_manage" in first
    conv.close()


def test_daily_child_stays_daily_when_its_profile_drops_web_fetch(tmp_path):
    # The skill-reviewer's tool list has no web_fetch; the session is still
    # a daily one (its prompt, its persisted meta).
    conv = Conversation(config=Config(), workdir=tmp_path, llm_client=FakeLLMClient(),
                        session_dir=tmp_path / "sessions", daily=True)
    assert conv.supervisor is not None
    conv.supervisor.plan([{"title": "t", "brief": "b", "profile": "skill-reviewer",
                           "context": "fork"}], created_by="user")
    task = conv.supervisor.board.live()[0]
    child, _ = conv.supervisor._open_agent(task)
    assert child.registry.get("web_fetch") is None
    assert child.daily_mode
    child.close()
    conv.close()
