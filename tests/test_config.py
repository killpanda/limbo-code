import os
import tempfile
from pathlib import Path

from limbo.config import (
    load_config,
    save_model_to_config,
)


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


# -- save_model_to_config (tomlkit write-back) ---------------------------------


def test_save_model_preserves_comments_and_other_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '# user notes\n[llm]\napi_key = "k"\nmodel = "deepseek-v4-pro"  # inline\n'
        '\n[tools]\nbash_enabled = false\n'
    )
    assert save_model_to_config("gpt-5.5", path) is True
    text = path.read_text()
    assert 'model = "gpt-5.5"' in text
    assert "# user notes" in text
    assert 'api_key = "k"' in text
    assert "bash_enabled = false" in text


