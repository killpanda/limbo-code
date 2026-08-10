import os
import tempfile
from pathlib import Path

import pytest

from limbo.config import (
    Config,
    KittyKeyboardMode,
    load_config,
    save_model_to_config,
)


def test_default_config():
    cfg = Config()
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.model == "deepseek-v4-pro"
    assert cfg.llm.max_iterations == 50
    assert cfg.tools.bash_enabled is True
    assert ".ssh" in cfg.safety.sensitive_files
    assert cfg.providers == {}


def test_load_config_providers_overrides(tmp_path):
    path = tmp_path / "providers.toml"
    path.write_text(
        """
[providers.codex]
base_url = "https://relay.example.com/v1"
api_key_env = "CODEX_API_KEY"

[providers.glm]
base_url = "https://api.z.ai/api/coding/paas/v4"
api_key = "glm-key"
headers = {x-relay = "on"}
"""
    )
    cfg = load_config(path)
    codex = cfg.providers["codex"]
    assert codex.base_url == "https://relay.example.com/v1"
    assert codex.api_key is None
    assert codex.api_key_env == "CODEX_API_KEY"
    assert codex.headers == {}
    glm = cfg.providers["glm"]
    assert glm.base_url == "https://api.z.ai/api/coding/paas/v4"
    assert glm.api_key == "glm-key"
    assert glm.api_key_env is None
    assert glm.headers == {"x-relay": "on"}


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
    assert cfg.llm.model == "deepseek-v4-pro"


def test_load_config_malformed_file_uses_defaults(tmp_path):
    path = tmp_path / "malformed.toml"
    path.write_text("[unclosed = ")
    with pytest.warns(UserWarning, match="Malformed config file"):
        cfg = load_config(path)
    assert cfg.llm.model == "deepseek-v4-pro"


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
    assert cfg.llm.model == "deepseek-v4-pro"


def test_load_config_validation_error_uses_defaults(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text('[llm]\nmax_iterations = "ten"\n')
    with pytest.warns(UserWarning, match="Invalid config file"):
        cfg = load_config(path)
    assert cfg.llm.max_iterations == 50


def test_llm_config_rejects_non_positive_max_iterations():
    from limbo.config import LLMConfig

    with pytest.raises(ValueError, match="max_iterations"):
        LLMConfig(max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        LLMConfig(max_iterations=-1)
    assert LLMConfig(max_iterations=1).max_iterations == 1


def test_llm_config_retry_defaults():
    from limbo.config import LLMConfig

    cfg = LLMConfig()
    assert cfg.max_retries == 3
    assert cfg.retry_base_delay == 1.0
    assert cfg.timeout == 600.0
    assert cfg.connect_timeout == 30.0


def test_load_config_retry_fields_from_toml(tmp_path):
    path = tmp_path / "retry.toml"
    path.write_text(
        "[llm]\n"
        "max_retries = 5\n"
        "retry_base_delay = 2.5\n"
        "timeout = 120.0\n"
        "connect_timeout = 10.0\n"
    )
    cfg = load_config(path)
    assert cfg.llm.max_retries == 5
    assert cfg.llm.retry_base_delay == 2.5
    assert cfg.llm.timeout == 120.0
    assert cfg.llm.connect_timeout == 10.0


def test_llm_config_clamps_negative_max_retries():
    from limbo.config import LLMConfig

    with pytest.warns(UserWarning, match="max_retries"):
        cfg = LLMConfig(max_retries=-1)
    assert cfg.max_retries == 0


def test_llm_config_clamps_non_positive_delays():
    from limbo.config import LLMConfig

    with pytest.warns(UserWarning, match="retry_base_delay"):
        cfg = LLMConfig(retry_base_delay=0)
    assert cfg.retry_base_delay == 1.0
    with pytest.warns(UserWarning, match="timeout"):
        cfg = LLMConfig(timeout=-5)
    assert cfg.timeout == 600.0
    with pytest.warns(UserWarning, match="connect_timeout"):
        cfg = LLMConfig(connect_timeout=0)
    assert cfg.connect_timeout == 30.0


def test_llm_config_connect_timeout_clamped_to_timeout():
    from limbo.config import LLMConfig

    with pytest.warns(UserWarning, match="connect_timeout"):
        cfg = LLMConfig(timeout=10.0, connect_timeout=60.0)
    assert cfg.connect_timeout == 10.0


def test_load_config_invalid_retry_value_does_not_discard_other_fields(tmp_path):
    """A bad retry_base_delay must clamp+warn, not reset the whole config."""
    path = tmp_path / "partial.toml"
    path.write_text('[llm]\nmodel = "gpt-4o"\nretry_base_delay = -1\n')
    with pytest.warns(UserWarning, match="retry_base_delay"):
        cfg = load_config(path)
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.retry_base_delay == 1.0


# -- save_model_to_config (tomlkit write-back) ---------------------------------


def test_save_model_creates_minimal_config(tmp_path):
    path = tmp_path / "sub" / "config.toml"
    assert save_model_to_config("glm-4.7", path) is True
    cfg = load_config(path)
    assert cfg.llm.model == "glm-4.7"


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


def test_save_model_malformed_file_returns_false_and_keeps_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[unclosed = ")
    assert save_model_to_config("glm-4.7", path) is False
    assert path.read_text() == "[unclosed = "


def test_save_model_unwritable_path_returns_false(tmp_path):
    path = tmp_path / "missing" / "config.toml"
    path.parent.mkdir  # sanity: parent helper exists but dir not created
    import os

    os.chmod(tmp_path, 0o500)
    try:
        assert save_model_to_config("glm-4.7", path) is False
    finally:
        os.chmod(tmp_path, 0o700)


def test_kitty_keyboard_default_is_auto():
    cfg = Config()
    assert cfg.ui.kitty_keyboard == KittyKeyboardMode.AUTO


def test_kitty_keyboard_mode_from_toml(tmp_path):
    path = tmp_path / "kitty.toml"
    path.write_text('[ui]\nkitty_keyboard = "disabled"\n')
    assert load_config(path).ui.kitty_keyboard is KittyKeyboardMode.DISABLED

    path.write_text('[ui]\nkitty_keyboard = "enabled"\n')
    assert load_config(path).ui.kitty_keyboard is KittyKeyboardMode.ENABLED


def test_kitty_keyboard_mode_disable_decision():
    # auto: disable only inside the herdr multiplexer (IME fix), keep the
    # kitty keyboard protocol everywhere else.
    assert KittyKeyboardMode.AUTO.disable_textual_kitty_key(in_herdr=True)
    assert not KittyKeyboardMode.AUTO.disable_textual_kitty_key(in_herdr=False)
    # enabled: never disable (opt out of the workaround).
    assert not KittyKeyboardMode.ENABLED.disable_textual_kitty_key(in_herdr=True)
    # disabled: always force raw/legacy input.
    assert KittyKeyboardMode.DISABLED.disable_textual_kitty_key(in_herdr=False)
