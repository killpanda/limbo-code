"""Tests for the CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import limbo.app as app_module
import limbo.ui.app as ui_app_module
from limbo.app import main
from limbo.config import Config


@pytest.fixture
def fake_config(monkeypatch):
    config = Config()
    config.llm.api_key = "fake-key"
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    return config


# --- session resume flags -------------------------------------------------


def _write_session(session_dir: Path, session_id: str, workdir: Path) -> Path:
    from limbo.sessions import SessionMeta, save_session

    path = session_dir / f"{session_id}.jsonl"
    save_session(
        path,
        SessionMeta(id=session_id, workdir=str(workdir.resolve())),
        [],
    )
    return path


def test_cli_resume_by_id_prefix_switches_workdir(
    fake_config, monkeypatch, tmp_path
):
    session_dir = tmp_path / "sessions"
    project = tmp_path / "elsewhere"
    project.mkdir()
    path = _write_session(session_dir, "def456", project)

    mock_app_class = MagicMock()
    monkeypatch.setattr(ui_app_module, "LimboApp", mock_app_class)
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "--resume", "def", "--session-dir", str(session_dir)],
    )

    assert main() == 0
    kwargs = mock_app_class.call_args.kwargs
    assert kwargs["resume"] == path
    # workdir follows the session's recorded directory.
    assert kwargs["workdir"] == project


# --- headless mode ------------------------------------------------------------


