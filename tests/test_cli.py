"""Tests for the CLI entry point."""

from __future__ import annotations

import os
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


def test_cli_passes_session_dir(fake_config, monkeypatch):
    mock_app_class = MagicMock()
    monkeypatch.setattr(ui_app_module, "LimboApp", mock_app_class)
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
    monkeypatch.setattr(ui_app_module, "LimboApp", mock_app_class)
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
    monkeypatch.setattr(ui_app_module, "LimboApp", mock_app_class)
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
    monkeypatch.setattr(ui_app_module, "LimboApp", MagicMock())
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


def test_cli_resume_unknown_id_fails(fake_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ui_app_module, "LimboApp", MagicMock())
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

    monkeypatch.setattr(ui_app_module, "LimboApp", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["limbo", "--resume", "abc", "--session-dir", str(session_dir)],
    )

    assert main() == 1
    err = capsys.readouterr().err
    assert "abc1" in err and "abc2" in err


def test_cli_model_flag_overrides_config(fake_config, monkeypatch):
    mock_app_class = MagicMock()
    monkeypatch.setattr(ui_app_module, "LimboApp", mock_app_class)
    monkeypatch.setattr(sys, "argv", ["limbo", "--model", "glm-4.7"])

    assert main() == 0

    assert fake_config.llm.model == "glm-4.7"
    kwargs = mock_app_class.call_args.kwargs
    assert kwargs["config"].llm.model == "glm-4.7"


def _probe_kitty_env(home: Path, herdr: bool) -> str:
    """Import limbo.app in a subprocess with a controlled HOME + HERDR_ENV and
    report whether TEXTUAL_DISABLE_KITTY_KEY ended up set."""
    import subprocess
    import sys

    probe = (
        "import os, sys; "
        "sys.argv = ['limbo']; "
        "import limbo.app; "
        "limbo.app.main(); "
        "print(os.environ.get('TEXTUAL_DISABLE_KITTY_KEY', ''))"
    )
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("TEXTUAL_DISABLE_KITTY_KEY", None)
    if herdr:
        env["HERDR_ENV"] = "1"
    else:
        env.pop("HERDR_ENV", None)
    run = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.strip()


def test_kitty_keyboard_config_drives_textual_env(tmp_path):
    """The `ui.kitty_keyboard` config option decides Textual's
    TEXTUAL_DISABLE_KITTY_KEY, not raw environment sniffing.

    Without the workaround, Textual's kitty keyboard REPORT_ALL_KEYS request
    makes herdr mirror report-all to the host terminal (Ghostty), and Ghostty
    then encodes IME commits as the physical commit key (space) while dropping
    the composed text — typing Chinese turns into spaces. Pasting is
    unaffected because it bypasses the key encoder.
    """
    home = tmp_path / "home"
    (home / ".limbo").mkdir(parents=True)
    config = home / ".limbo" / "config.toml"

    # auto (default) + inside herdr -> disable (the IME fix)
    config.write_text('[ui]\nkitty_keyboard = "auto"\n')
    assert _probe_kitty_env(home, herdr=True) == "1"
    # auto + direct terminal -> keep the kitty keyboard protocol
    assert _probe_kitty_env(home, herdr=False) == ""
    # enabled + herdr -> user opted out of the workaround
    config.write_text('[ui]\nkitty_keyboard = "enabled"\n')
    assert _probe_kitty_env(home, herdr=True) == ""
    # disabled everywhere
    config.write_text('[ui]\nkitty_keyboard = "disabled"\n')
    assert _probe_kitty_env(home, herdr=False) == "1"


def test_kitty_keyboard_config_defaults_to_auto(tmp_path):
    """Without a config file the default is `auto`: inside herdr the
    workaround is on, elsewhere the kitty keyboard protocol stays enabled."""
    home = tmp_path / "home"
    (home / ".limbo").mkdir(parents=True)
    assert _probe_kitty_env(home, herdr=True) == "1"
    assert _probe_kitty_env(home, herdr=False) == ""
