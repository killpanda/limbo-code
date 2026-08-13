"""Tests for the slash-command registry (limbo.ui.commands)."""

from __future__ import annotations

from pathlib import Path

from limbo.skills import Skill
from limbo.ui.commands import SlashCommand, SlashCommandRegistry


def make_skill(name: str, description: str = "") -> Skill:
    return Skill(name=name, description=description, body="", path=Path("/x"))


def test_register_and_get():
    registry = SlashCommandRegistry()
    calls = []
    registry.register(
        SlashCommand("/help", "显示帮助", handler=lambda arg: calls.append(arg))
    )

    command = registry.get("/help")
    assert command is not None
    command.handler("arg1")
    assert calls == ["arg1"]
    assert registry.get("/nope") is None


def test_all_returns_registered_commands():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/a", "first"))
    registry.register(SlashCommand("/b", "second"))

    assert [c.name for c in registry.all()] == ["/a", "/b"]


def test_candidates_merges_skills_after_builtins():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/new", "开始新会话"))

    candidates = registry.candidates([make_skill("tdd", "TDD workflow")])

    names = [c.name for c in candidates]
    assert names == ["/new", "/tdd"]
    skill_command = candidates[1]
    assert skill_command.kind == "skill"
    assert skill_command.takes_args is True
    assert "[skill]" in skill_command.description


def test_builtin_shadows_same_named_skill():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/new", "开始新会话"))

    candidates = registry.candidates([make_skill("new", "a skill named new")])

    assert len(candidates) == 1
    assert candidates[0].kind == "builtin"


def test_help_text_lists_commands():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/sessions", "切换历史会话"))
    registry.register(SlashCommand("/new", "开始新会话"))

    text = registry.help_text()
    assert "/sessions" in text
    assert "/new" in text


# -- resolve(): the single collision-policy site (menu + dispatch) ------------


def test_resolve_returns_builtin_and_skill():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/new", "开始新会话"))
    skill = make_skill("tdd")

    assert registry.resolve("/new", [skill]) is registry.get("/new")
    assert registry.resolve("/tdd", [skill]) is skill
    assert registry.resolve("/nope", [skill]) is None


def test_resolve_builtin_wins_collision_with_skill():
    """Dispatch and menu share one policy: the built-in claims the name."""
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/new", "开始新会话"))
    skill = make_skill("new", "a skill named new")

    resolved = registry.resolve("/new", [skill])
    assert isinstance(resolved, SlashCommand)
    assert resolved.kind == "builtin"
    # Menu side agrees (no skill entry for the shadowed name).
    assert [c.name for c in registry.candidates([skill])] == ["/new"]


def test_resolve_builtin_lookup_is_case_insensitive():
    registry = SlashCommandRegistry()
    registry.register(SlashCommand("/help", "显示帮助"))

    assert registry.resolve("/HELP", []) is registry.get("/help")
    # Skill names stay exact-case.
    skill = make_skill("tdd")
    assert registry.resolve("/TDD", [skill]) is None
