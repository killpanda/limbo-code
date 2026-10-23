"""Shift+Enter inside herdr: request only the kitty disambiguate flag."""

from __future__ import annotations

import inspect
import os

import pytest
from textual.drivers import linux_driver

from limbo.app import _apply_kitty_keyboard_workaround
from limbo.config import Config, KittyKeyboardMode


@pytest.fixture
def restore_driver_flags(monkeypatch):
    for name in ("KITTY_REPORT_ALL_KEYS", "KITTY_REPORT_ASSOCIATED_TEXT"):
        monkeypatch.setattr(linux_driver, name, getattr(linux_driver, name))
    monkeypatch.delenv("TEXTUAL_DISABLE_KITTY_KEY", raising=False)


def requested_flags() -> int:
    return (
        linux_driver.KITTY_DISAMBIGUATE_ESCAPE_CODES
        | linux_driver.KITTY_REPORT_ALL_KEYS
        | linux_driver.KITTY_REPORT_ASSOCIATED_TEXT
    )


def run_workaround(monkeypatch, mode: KittyKeyboardMode, *, herdr: bool) -> None:
    if herdr:
        monkeypatch.setenv("HERDR_ENV", "1")
    else:
        monkeypatch.delenv("HERDR_ENV", raising=False)
    cfg = Config()
    cfg.ui.kitty_keyboard = mode
    _apply_kitty_keyboard_workaround(cfg)


def test_herdr_requests_disambiguate_only(monkeypatch, restore_driver_flags):
    # Disabling the protocol made herdr send Shift+Enter as a bare "\r";
    # the full flag set breaks IME. Only ">1u" gets both right.
    run_workaround(monkeypatch, KittyKeyboardMode.AUTO, herdr=True)
    assert requested_flags() == 1
    assert "TEXTUAL_DISABLE_KITTY_KEY" not in os.environ


def test_outside_herdr_keeps_full_request(monkeypatch, restore_driver_flags):
    run_workaround(monkeypatch, KittyKeyboardMode.AUTO, herdr=False)
    assert requested_flags() == 0b11001


def test_disabled_mode_turns_protocol_off(monkeypatch, restore_driver_flags):
    run_workaround(monkeypatch, KittyKeyboardMode.DISABLED, herdr=True)
    assert os.environ["TEXTUAL_DISABLE_KITTY_KEY"] == "1"


def test_textual_still_builds_request_from_module_flags():
    # The patch relies on Textual reading these globals at call time.
    source = inspect.getsource(linux_driver.LinuxDriver.start_application_mode)
    assert "KITTY_REPORT_ALL_KEYS" in source
    assert "KITTY_REPORT_ASSOCIATED_TEXT" in source
