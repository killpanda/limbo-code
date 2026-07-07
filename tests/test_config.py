import os
import tempfile
from pathlib import Path

import pytest

from limbo.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.model == "deepseek-chat"
    assert cfg.llm.max_iterations == 10
    assert cfg.tools.bash_enabled is True
    assert ".ssh" in cfg.safety.sensitive_files


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('''
[llm]
api_key = "test-key"
model = "gpt-4o"
''')
        path = f.name
    try:
        cfg = load_config(Path(path))
        assert cfg.llm.api_key == "test-key"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    finally:
        os.unlink(path)


def test_load_config_missing_file_uses_defaults():
    cfg = load_config(Path("/nonexistent/config.toml"))
    assert cfg.llm.model == "deepseek-chat"


def test_load_config_malformed_file_uses_defaults(tmp_path):
    path = tmp_path / "malformed.toml"
    path.write_text("[unclosed = ")
    with pytest.warns(UserWarning, match="Malformed config file"):
        cfg = load_config(path)
    assert cfg.llm.model == "deepseek-chat"


def test_load_config_tools_bash_enabled(tmp_path):
    path = tmp_path / "tools.toml"
    path.write_text("[tools]\nbash_enabled = false\n")
    cfg = load_config(path)
    assert cfg.tools.bash_enabled is False


def test_load_config_permission_error_uses_defaults(tmp_path):
    path = tmp_path / "unreadable.toml"
    path.write_text("[llm]\nmodel = \"gpt-4o\"\n")
    path.chmod(0o000)
    try:
        with pytest.warns(UserWarning, match="Could not read config file"):
            cfg = load_config(path)
    finally:
        path.chmod(0o644)
    assert cfg.llm.model == "deepseek-chat"
