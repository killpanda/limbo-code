"""Tests for Code Mode presentation: SDK rendering and the system prompt."""

import tempfile
from pathlib import Path

import pytest

from limbo.code_mode import CODE_ONLY_INSTRUCTION, render_tools_sdk
from limbo.prompt import build_system_prompt
from limbo.tools.registry import ToolRegistry


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def registry(workdir):
    return ToolRegistry(workdir=workdir)


def test_sdk_lists_tools_with_signatures(registry):
    sdk = render_tools_sdk(registry)
    # Every visible tool appears as an awaitable method.
    assert "tools.read(" in sdk
    assert "tools.edit(" in sdk
    assert "tools.write(" in sdk
    # run_code itself must never be listed (no nested runs).
    assert "tools.run_code(" not in sdk
    # Required parameters have no default; optional ones do.
    assert "path: str" in sdk  # read requires path
    assert "offset: int = None" in sdk  # read offset is optional


def test_sdk_respects_bash_disabled(workdir):
    from limbo.config import Config

    cfg = Config()
    cfg.tools.bash_enabled = False
    reg = ToolRegistry(workdir=workdir, config=cfg)
    sdk = render_tools_sdk(reg)
    assert "tools.bash(" not in sdk


def test_sdk_deterministic(registry):
    assert render_tools_sdk(registry) == render_tools_sdk(registry)


def test_system_prompt_native_mode_unchanged(registry, workdir):
    prompt = build_system_prompt(registry, workdir)
    assert "run_code" not in prompt
    assert CODE_ONLY_INSTRUCTION not in prompt
    assert "- read: " in prompt


def test_system_prompt_code_mode(registry, workdir):
    registry.code_mode = True
    prompt = build_system_prompt(registry, workdir)
    # The collapse statement and SDK section are injected.
    assert CODE_ONLY_INSTRUCTION in prompt
    assert "## Writing code for run_code" in prompt
    assert "tools.read(" in prompt
    # The native tool list collapses to run_code only.
    assert "- run_code: " in prompt
    assert "- read: " not in prompt

def test_sdk_teaches_batching_not_drip_feeding(registry):
    """Code Mode's value is orchestration: the model-facing text must push
    batching many tool calls into one program, not wrapping single calls
    (regression guard for observed one-command-per-run_code behavior)."""
    from limbo.tools.run_code import RunCodeTool

    sdk = render_tools_sdk(registry)
    assert "round trip" in sdk
    assert "round trip" in RunCodeTool.description
