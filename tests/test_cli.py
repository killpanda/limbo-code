"""Tests for the CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import limbo.app as app_module
from limbo.app import main
from limbo.config import Config


@pytest.fixture
def fake_config(monkeypatch):
    config = Config()
    config.llm.api_key = "fake-key"
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    return config


def test_cli_passes_session_dir(fake_config, monkeypatch):
    mock_app_class = MagicMock()
    monkeypatch.setattr(app_module, "LimboApp", mock_app_class)
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "--session-dir", "/tmp/sessions", "--workdir", "/tmp/project"],
    )

    assert main() == 0

    mock_app_class.assert_called_once()
    kwargs = mock_app_class.call_args.kwargs
    assert kwargs["session_dir"] == Path("/tmp/sessions")
    assert kwargs["workdir"] == Path("/tmp/project")


def test_cli_defaults_session_dir_to_none(fake_config, monkeypatch):
    mock_app_class = MagicMock()
    monkeypatch.setattr(app_module, "LimboApp", mock_app_class)
    monkeypatch.setattr(sys, "argv", ["limbo"])

    assert main() == 0

    assert mock_app_class.call_args.kwargs["session_dir"] is None


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


def test_cli_continue_resumes_latest_for_workdir(fake_config, monkeypatch, tmp_path):
    session_dir = tmp_path / "sessions"
    project = tmp_path / "project"
    project.mkdir()
    path = _write_session(session_dir, "abc123", project)

    mock_app_class = MagicMock()
    monkeypatch.setattr(app_module, "LimboApp", mock_app_class)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "--continue",
            "--workdir",
            str(project),
            "--session-dir",
            str(session_dir),
        ],
    )

    assert main() == 0
    kwargs = mock_app_class.call_args.kwargs
    assert kwargs["resume"] == path
    assert kwargs["workdir"] == project


def test_cli_continue_without_sessions_fails(fake_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app_module, "LimboApp", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limbo",
            "--continue",
            "--workdir",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
        ],
    )

    assert main() == 1
    assert "no session" in capsys.readouterr().err.lower()


def test_cli_resume_by_id_prefix_switches_workdir(
    fake_config, monkeypatch, tmp_path
):
    session_dir = tmp_path / "sessions"
    project = tmp_path / "elsewhere"
    project.mkdir()
    path = _write_session(session_dir, "def456", project)

    mock_app_class = MagicMock()
    monkeypatch.setattr(app_module, "LimboApp", mock_app_class)
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


def test_cli_resume_unknown_id_fails(fake_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app_module, "LimboApp", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "--resume", "nope", "--session-dir", str(tmp_path)],
    )

    assert main() == 1
    assert "no session" in capsys.readouterr().err.lower()


def test_cli_resume_ambiguous_id_lists_candidates(
    fake_config, monkeypatch, tmp_path, capsys
):
    session_dir = tmp_path / "sessions"
    _write_session(session_dir, "abc1", tmp_path)
    _write_session(session_dir, "abc2", tmp_path)

    monkeypatch.setattr(app_module, "LimboApp", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "--resume", "abc", "--session-dir", str(session_dir)],
    )

    assert main() == 1
    err = capsys.readouterr().err
    assert "abc1" in err and "abc2" in err
