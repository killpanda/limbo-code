"""Tests for the dialect-client scaffolding (shared plumbing helpers)."""

from __future__ import annotations

import base64

import pytest

from limbo.config import Config
from limbo.llm.catalog import resolve_model
from limbo.llm.scaffold import (
    encode_image_data,
    http_timeout,
    map_thinking_effort,
    require_api_key,
)
from limbo.models import Attachment


def test_require_api_key_returns_configured_key():
    config = Config()
    config.llm.api_key = "sk-test"
    assert require_api_key(resolve_model("deepseek-chat"), config) == "sk-test"


def test_require_api_key_error_names_provider_and_env():
    config = Config()
    config.llm.api_key = None
    with pytest.raises(ValueError, match=r"No API key for provider 'deepseek'"):
        require_api_key(resolve_model("deepseek-chat"), config)
    with pytest.raises(ValueError, match=r"\$DEEPSEEK_API_KEY"):
        require_api_key(resolve_model("deepseek-chat"), config)


def test_require_api_key_generic_provider_has_no_env_hint():
    config = Config()
    config.llm.api_key = None
    try:
        require_api_key(resolve_model("some-unknown-model"), config)
    except ValueError as e:
        assert "environment variable" not in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_http_timeout_uses_config():
    config = Config()
    config.llm.timeout = 123.0
    config.llm.connect_timeout = 4.0
    timeout = http_timeout(config)
    assert timeout.read == 123.0
    assert timeout.connect == 4.0


def test_map_thinking_effort_maps_known_level():
    spec = resolve_model("kimi-k3")
    assert map_thinking_effort(spec, "high") == "high"


def test_map_thinking_effort_warns_and_returns_none_on_unknown():
    spec = resolve_model("kimi-k3")
    with pytest.warns(UserWarning, match="not supported by kimi-k3"):
        assert map_thinking_effort(spec, "ludicrous") is None


def test_encode_image_data_roundtrip(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n")
    attachment = Attachment(kind="image", name="shot.png", path=str(image))
    data, mime = encode_image_data(attachment)
    assert base64.standard_b64decode(data) == b"\x89PNG\r\n"
    assert mime == "image/png"


def test_encode_image_data_missing_file_degrades_to_none(tmp_path):
    attachment = Attachment(
        kind="image", name="gone.png", path=str(tmp_path / "gone.png")
    )
    assert encode_image_data(attachment) is None


def test_encode_image_data_honors_explicit_mime(tmp_path):
    image = tmp_path / "shot.jpg"
    image.write_bytes(b"\xff\xd8")
    attachment = Attachment(
        kind="image", name="shot.jpg", path=str(image), mime="image/jpeg"
    )
    _, mime = encode_image_data(attachment)
    assert mime == "image/jpeg"
