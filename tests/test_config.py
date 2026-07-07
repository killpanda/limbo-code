import os
import tempfile
from pathlib import Path

from limbo.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.model == "deepseek-chat"
    assert cfg.llm.max_iterations == 10


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
