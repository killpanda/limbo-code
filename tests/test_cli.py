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
