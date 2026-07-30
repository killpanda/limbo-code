"""Suite-wide guards: tests must never touch the developer's real ~/.limbo."""

from __future__ import annotations

import pytest

import limbo.config as config_module


@pytest.fixture(autouse=True)
def isolated_default_config_path(tmp_path, monkeypatch):
    """Redirect the default config.toml to a per-test location.

    Regression guard: /model persists via save_model_to_config(), which
    defaults to the real ~/.limbo/config.toml. Tests that exercised the
    swap path without isolation rewrote the developer's real config —
    resetting their model to the default on the next limbo launch.
    """
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml"
    )
