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


@pytest.fixture(autouse=True)
def isolated_omarchy_state(tmp_path, monkeypatch):
    """Point Omarchy theme detection at an empty per-test directory.

    Developers running the suite on an Omarchy desktop would otherwise get
    the ``omarchy`` theme instead of limbo-dusk, changing snapshots and
    theme assertions depending on their wallpaper of the day.
    """
    import limbo.ui.omarchy as omarchy_module

    monkeypatch.setattr(
        omarchy_module, "DEFAULT_STATE_DIR", tmp_path / "omarchy-state"
    )


@pytest.fixture(autouse=True)
def motion_off(monkeypatch):
    """Animations render their final frame in tests (Limbo Motion ``off``).

    Tests that exercise motion itself pass an explicit mode or env mapping.
    """
    monkeypatch.setenv("LIMBO_MOTION", "off")


@pytest.fixture(autouse=True)
def isolated_auth_file(tmp_path, monkeypatch):
    """Stored logins (``limbo auth login``) live in a per-test auth.json.

    API-key resolution reads ~/.limbo/auth.json; without this a developer's
    real logins would leak into "no key configured" tests.
    """
    monkeypatch.delenv("LIMBO_AUTH_FILE", raising=False)
    import limbo.auth as auth_module

    monkeypatch.setattr(auth_module, "DEFAULT_AUTH_PATH", tmp_path / "auth.json")


@pytest.fixture(autouse=True)
def isolated_deepseek_key(monkeypatch):
    """A developer's $DEEPSEEK_API_KEY must not add web_search to every registry."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
