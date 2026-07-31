"""Tests for the headless `limbo exec` mode."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

import limbo.app as app_module
import limbo.exec as exec_module
from limbo.app import main
from limbo.config import Config
from limbo.exec import run_headless
from limbo.models import TextChunk


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools, on_request=None):
        for event in self.responses.pop(0):
            yield event


@pytest.fixture
def fake_config(monkeypatch):
    config = Config()
    config.llm.api_key = "fake-key"
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    return config


# --- run_headless ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_headless_streams_text_to_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        exec_module,
        "create_llm_client",
        lambda config: FakeLLMClient([[TextChunk(text="hello")]]),
    )

    code = await run_headless(
        "hi", config=Config(), workdir=tmp_path, session_dir=tmp_path / "s"
    )

    assert code == 0
    out, err = capsys.readouterr()
    assert out == "hello"
    assert "exec start" in err
    assert "exec end" in err


@pytest.mark.asyncio
async def test_run_headless_returns_1_on_error(tmp_path, monkeypatch, capsys):
    class FailingLLMClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover - make this an async generator

    monkeypatch.setattr(
        exec_module, "create_llm_client", lambda config: FailingLLMClient()
    )

    code = await run_headless(
        "hi", config=Config(), workdir=tmp_path, session_dir=tmp_path / "s"
    )

    assert code == 1
    _, err = capsys.readouterr()
    assert "ERROR" in err


# --- CLI wiring ------------------------------------------------------------


def _patch_run_headless(monkeypatch) -> AsyncMock:
    # main() wraps the call in asyncio.run(), so the mock must be awaitable.
    mock = AsyncMock(return_value=0)
    monkeypatch.setattr(app_module, "run_headless", mock)
    return mock


def test_exec_requires_prompt(fake_config, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["limbo", "exec"])
    with pytest.raises(SystemExit):
        main()


def test_exec_reads_prompt_file(fake_config, monkeypatch, tmp_path):
    mock = _patch_run_headless(monkeypatch)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("fix the bug", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "exec",
            "--workdir",
            str(tmp_path),
            "--prompt-file",
            str(prompt_file),
        ],
    )

    assert main() == 0

    mock.assert_called_once()
    assert mock.call_args.args[0] == "fix the bug"
    assert mock.call_args.kwargs["workdir"] == tmp_path.resolve()
    assert mock.call_args.kwargs["session_dir"] == app_module.DEFAULT_EXEC_SESSION_DIR


def test_exec_accepts_inline_prompt_and_session_dir(
    fake_config, monkeypatch, tmp_path
):
    mock = _patch_run_headless(monkeypatch)
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "exec",
            "--workdir",
            str(tmp_path),
            "--prompt",
            "inline prompt",
            "--session-dir",
            str(session_dir),
        ],
    )

    assert main() == 0
    assert mock.call_args.args[0] == "inline prompt"
    assert mock.call_args.kwargs["session_dir"] == session_dir


def test_exec_rejects_missing_prompt_file(fake_config, monkeypatch, tmp_path, capsys):
    _patch_run_headless(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "exec",
            "--workdir",
            str(tmp_path),
            "--prompt-file",
            str(tmp_path / "nope.txt"),
        ],
    )

    assert main() == 1
    assert "Cannot read prompt file" in capsys.readouterr().err


def test_exec_rejects_missing_workdir(fake_config, monkeypatch, tmp_path, capsys):
    _patch_run_headless(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "exec", "--workdir", str(tmp_path / "nope"), "--prompt", "hi"],
    )

    assert main() == 1
    assert "workdir does not exist" in capsys.readouterr().err


def test_exec_model_override(fake_config, monkeypatch, tmp_path):
    mock = _patch_run_headless(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "exec",
            "--workdir",
            str(tmp_path),
            "--prompt",
            "hi",
            "--model",
            "glm-4.7",
        ],
    )

    assert main() == 0
    assert mock.call_args.kwargs["config"].llm.model == "glm-4.7"
