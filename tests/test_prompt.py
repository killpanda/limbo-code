"""Tests for system-prompt assembly.

The registry is the single source of truth for the tools section: a tool
that is not registered (e.g. bash_enabled=false) never appears in the
prompt, and descriptions are not duplicated by hand.
"""

from __future__ import annotations

from pathlib import Path

from limbo.config import Config
from limbo.prompt import build_system_prompt
from limbo.skills import Skill
from limbo.tools.registry import ToolRegistry


def _registry(workdir: Path, config: Config | None = None) -> ToolRegistry:
    return ToolRegistry(workdir=workdir, config=config or Config())


def test_tools_section_lists_registered_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    prompt = build_system_prompt(_registry(tmp_path), tmp_path)
    for name in ("read", "bash", "edit", "write", "grep", "find", "ls"):
        assert f"- {name}: " in prompt


def test_tools_section_derives_descriptions_from_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    registry = _registry(tmp_path)
    prompt = build_system_prompt(registry, tmp_path)
    # The prompt line carries the tool's own description (first sentence),
    # not a hand-maintained summary.
    first_sentence = registry.get("ls").description.split(". ")[0]
    assert f"- ls: {first_sentence}" in prompt


def test_disabled_bash_never_appears(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    config = Config()
    config.tools.bash_enabled = False
    prompt = build_system_prompt(_registry(tmp_path, config), tmp_path)
    assert "- bash:" not in prompt
    # The guidelines stop pointing at a tool that does not exist.
    assert "over bash" not in prompt


def test_bash_guideline_present_when_bash_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    prompt = build_system_prompt(_registry(tmp_path), tmp_path)
    assert "over bash" in prompt


def test_wraps_agents_md_in_xml_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    (tmp_path / "AGENTS.md").write_text("# Project rules\n")
    home = tmp_path / "home"
    (home / ".limbo").mkdir(parents=True)
    (home / ".limbo" / "AGENTS.md").write_text("# Global rules\n")
    prompt = build_system_prompt(_registry(tmp_path), tmp_path, home=home)
    assert "<project_context>" in prompt
    assert f'<project_instructions path="{tmp_path / "AGENTS.md"}">' in prompt
    assert "# Project rules" in prompt
    assert "# Global rules" in prompt
    assert "</project_context>" in prompt


def test_includes_skills_catalog(tmp_path, monkeypatch):
    skill = Skill(
        name="tdd",
        description="Test-driven development.",
        body="# TDD",
        path=tmp_path / "SKILL.md",
    )
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [skill])
    prompt = build_system_prompt(_registry(tmp_path), tmp_path)
    assert "<available_skills>" in prompt
    assert "tdd" in prompt


def test_skips_catalog_when_no_skills(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir: [])
    prompt = build_system_prompt(_registry(tmp_path), tmp_path)
    assert "<available_skills>" not in prompt
