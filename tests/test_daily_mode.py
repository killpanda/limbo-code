"""Daily mode: the posture toggle, persistent memory, web_fetch."""

from __future__ import annotations

import httpx
import pytest
import respx

from limbo import memory as memory_module
from limbo.agent import Agent
from limbo.config import Config
from limbo.conversation import Conversation
from limbo.headless import SlashDaily, SlashMemory, parse_submission
from limbo.memory import MemoryStore, MemoryStoreError, parse_memory_command
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent
from limbo.tools.registry import ToolRegistry
from limbo.tools.web_fetch import WebFetchTool, html_to_text


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.tools_seen: list[list[str]] = []

    async def chat(self, messages, tools, on_request=None):
        self.tools_seen.append([t["function"]["name"] for t in tools])
        for event in self.responses.pop(0):
            yield event


def _agent(tmp_path, responses=None, **kwargs) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient(responses),
        workdir=tmp_path,
        session_dir=tmp_path / "sessions",
        **kwargs,
    )


def _store(tmp_path, user=200, notes=200) -> MemoryStore:
    return MemoryStore(tmp_path / "mem", {"user": user, "notes": notes})


# -- memory store -------------------------------------------------------------


def test_memory_batch_add_patch_remove(tmp_path):
    store = _store(tmp_path)
    store.apply("user", [{"action": "add", "content": "Name: Jack"},
                         {"action": "add", "content": "Prefers Chinese replies"}])
    store.apply("user", [{"action": "patch", "old_text": "Jack", "content": "Jack Mini"},
                         {"action": "remove", "old_text": "Chinese"}])
    assert store.entries("user") == ["Name: Jack Mini"]
    # The previous version is kept for one step of undo.
    assert "Chinese" in (tmp_path / "mem" / "USER.md.bak").read_text()


def test_memory_limit_is_checked_on_the_final_result_and_is_atomic(tmp_path):
    store = _store(tmp_path, user=120)
    store.apply("user", [{"action": "add", "content": "a" * 100}])
    with pytest.raises(MemoryStoreError, match="limit is 120"):
        store.apply("user", [{"action": "add", "content": "b" * 50}])
    assert store.entries("user") == ["a" * 100]
    # Freeing room in the same batch makes the add fit.
    store.apply("user", [{"action": "remove", "old_text": "aaa"},
                         {"action": "add", "content": "b" * 50}])
    assert store.entries("user") == ["b" * 50]


def test_memory_ambiguous_old_text_writes_nothing(tmp_path):
    store = _store(tmp_path)
    store.apply("notes", [{"action": "add", "content": "uses vim"},
                          {"action": "add", "content": "uses zsh"}])
    with pytest.raises(MemoryStoreError, match="2 entries contain"):
        store.apply("notes", [{"action": "remove", "old_text": "uses"}])
    assert len(store.entries("notes")) == 2


def test_memory_command_parsing():
    assert parse_memory_command("") is None
    assert parse_memory_command("edit") == "user"
    assert parse_memory_command("edit notes") == "notes"
    with pytest.raises(ValueError):
        parse_memory_command("edit other")
    assert parse_submission("/daily-mode on") == SlashDaily("on")
    assert parse_submission("/memory edit notes") == SlashMemory("edit notes")


# -- posture toggle -------------------------------------------------------------


def test_daily_tools_come_and_go_with_the_mode(tmp_path):
    registry = ToolRegistry(tmp_path, Config())
    assert registry.get("web_fetch") is None and registry.get("memory") is None
    registry.daily_mode = True
    assert registry.get("web_fetch") is not None and registry.get("memory") is not None
    registry.daily_mode = False
    assert registry.get("web_fetch") is None and registry.get("memory") is None


def test_memory_off_keeps_web_fetch_but_no_memory_tool(tmp_path):
    config = Config()
    config.daily.memory = False
    registry = ToolRegistry(tmp_path, config)
    registry.daily_mode = True
    assert registry.get("web_fetch") is not None
    assert registry.get("memory") is None and registry.memory_store is None


def test_prompt_follows_mode_and_memory_snapshot_is_frozen(tmp_path):
    store = MemoryStore(memory_module.memory_dir(), {"user": 500, "notes": 500})
    store.apply("user", [{"action": "add", "content": "Name: Jack"}])
    store.apply("notes", [{"action": "add", "content": "Notes live in ~/notes"}])
    agent = _agent(tmp_path)
    coding = agent.messages[0].content or ""
    assert "coding assistant" in coding
    assert "Name: Jack" in coding  # USER.md rides in coding sessions too
    assert "~/notes" not in coding  # NOTES.md only in daily mode

    assert agent.set_daily_mode(True)
    daily = agent.messages[0].content or ""
    assert "personal assistant" in daily and "~/notes" in daily
    # A write mid-session never changes this session's prompt.
    store.apply("user", [{"action": "add", "content": "Likes tea"}])
    agent.set_daily_mode(False)
    agent.set_daily_mode(True)
    assert "Likes tea" not in (agent.messages[0].content or "")
    assert not agent.set_daily_mode(True)


def test_user_profile_can_be_kept_out_of_coding_sessions(tmp_path):
    MemoryStore(memory_module.memory_dir(), {"user": 500}).apply(
        "user", [{"action": "add", "content": "Name: Jack"}]
    )
    config = Config()
    config.daily.inject_user_in_coding = False
    agent = Agent(config=config, llm_client=FakeLLMClient(), workdir=tmp_path,
                  session_dir=tmp_path / "sessions")
    assert "Name: Jack" not in (agent.messages[0].content or "")


@pytest.mark.asyncio
async def test_daily_mode_persists_across_resume_and_memory_tool_runs(tmp_path):
    agent = _agent(
        tmp_path,
        [
            [ToolCallEvent(id="c1", name="memory", arguments={
                "target": "user",
                "operations": [{"action": "add", "content": "Lives in Shanghai"}],
            }), CompletionMeta()],
            [TextChunk(text="noted"), CompletionMeta()],
        ],
        meta_fields={"daily_mode": True},
    )
    assert agent.daily_mode
    _ = [e async for e in agent.run("remember I live in Shanghai")]
    assert "memory" in agent.llm_client.tools_seen[0]
    assert MemoryStore(memory_module.memory_dir(), {}).entries("user") == ["Lives in Shanghai"]

    resumed = _agent(tmp_path, resume=agent.session_path)
    assert resumed.daily_mode
    assert "personal assistant" in (resumed.messages[0].content or "")
    assert "Lives in Shanghai" in (resumed.messages[0].content or "")


def test_new_session_follows_the_current_mode(tmp_path):
    conv = Conversation(config=Config(), workdir=tmp_path, llm_client=FakeLLMClient(),
                        session_dir=tmp_path / "sessions", daily=True)
    assert conv.agent.daily_mode
    conv.new()
    assert conv.agent.daily_mode
    conv.agent.set_daily_mode(False)
    conv.new()
    assert not conv.agent.daily_mode
    conv.close()


def _skill(root, name, description):
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nbody\n"
    )


def test_daily_skills_are_seen_only_in_daily_mode(tmp_path):
    import limbo.skills as skills_module

    _skill(tmp_path / ".agents" / "skills", "tdd", "Test-driven development.")
    _skill(skills_module.DEFAULT_DAILY_SKILLS_DIR, "weekly-review", "Plan the week.")
    agent = _agent(tmp_path)
    coding = agent.messages[0].content or ""
    assert "tdd" in coding and "weekly-review" not in coding
    agent.set_daily_mode(True)
    daily = agent.messages[0].content or ""
    assert "tdd" in daily and "weekly-review" in daily


# -- web_fetch --------------------------------------------------------------------

PAGE = """<html><head><title>Tea Guide</title><script>var x=1;</script></head>
<body><nav><a href="/">Home</a></nav>
<article><h1>Green tea</h1><p>Brew at <b>80°C</b> for
2 minutes.</p><ul><li>Sencha</li><li><a href="/longjing">Longjing</a></li></ul></article>
<footer>© 2026</footer></body></html>"""


def test_html_to_text_keeps_the_article():
    title, text = html_to_text(PAGE, "https://tea.example/guide")
    assert title == "Tea Guide"
    assert "# Green tea" in text
    assert "Brew at 80°C for 2 minutes." in text
    assert "- [Longjing](https://tea.example/longjing)" in text
    assert "var x" not in text and "Home" not in text and "2026" not in text


@respx.mock
def test_web_fetch_pages_long_text(tmp_path):
    respx.get("https://tea.example/long").mock(
        return_value=httpx.Response(200, text="x" * 250, headers={"content-type": "text/plain"})
    )
    tool = WebFetchTool(tmp_path, max_chars=100)
    first = tool.execute({"url": "https://tea.example/long"})
    assert first.success and "offset=100" in (first.output or "")
    last = tool.execute({"url": "https://tea.example/long", "offset": 200})
    assert "Characters 200-250 of 250.]" in (last.output or "")


@respx.mock
def test_web_fetch_refuses_pdf_and_reports_http_errors(tmp_path):
    respx.get("https://x.example/a.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.7 ...",
                                    headers={"content-type": "application/pdf"})
    )
    respx.get("https://x.example/missing").mock(return_value=httpx.Response(404))
    tool = WebFetchTool(tmp_path)
    assert "PDF" in (tool.execute({"url": "https://x.example/a.pdf"}).error or "")
    assert "HTTP 404" in (tool.execute({"url": "https://x.example/missing"}).error or "")
    assert not tool.execute({"url": "file:///etc/passwd"}).success
