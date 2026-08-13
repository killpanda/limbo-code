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


@pytest.fixture(autouse=True)
def isolated_herdr_env(monkeypatch):
    """Strip Herdr pane variables from the test process environment.

    Running the suite inside a Herdr pane (e.g. ``make check`` in one)
    inherits HERDR_ENV=1 plus the pane id / socket, so every UI test that
    instantiates MainScreen would create a real HerdrReporter and report
    working/idle/release for the pane — clobbering Herdr's view of the
    REAL agent running in that pane. Tests that exercise the integration
    pass an explicit env dict to from_env()/create_reporters() and never
    read os.environ, so stripping here is invisible to them.
    """
    for name in (
        "HERDR_ENV",
        "HERDR_PANE_ID",
        "HERDR_BIN_PATH",
        "HERDR_SOCKET_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
