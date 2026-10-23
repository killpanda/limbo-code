"""Tests for system-prompt assembly.

The registry is the single source of truth for the tools section: a tool
that is not registered (e.g. bash_enabled=false) never appears in the
prompt, and descriptions are not duplicated by hand.
"""

from __future__ import annotations

from pathlib import Path

from limbo.config import Config
from limbo.prompt import build_system_prompt
from limbo.tools.registry import ToolRegistry


def _registry(workdir: Path, config: Config | None = None) -> ToolRegistry:
    return ToolRegistry(workdir=workdir, config=config or Config())


def test_tools_section_lists_registered_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.prompt.discover_skills", lambda workdir, **_: [])
    prompt = build_system_prompt(_registry(tmp_path), tmp_path)
    for name in ("read", "bash", "edit", "write", "grep", "find", "ls"):
        assert f"- {name}: " in prompt


