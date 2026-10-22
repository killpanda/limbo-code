"""Stored provider credentials (``~/.limbo/auth.json``), pi/opencode style.

``limbo auth login`` and the TUI's model dialog write API keys here instead
of making users edit config.toml or export env vars. The file is created
0600 and holds one entry per catalog provider id::

    {"kimi-coding": {"type": "api", "key": "sk-...", "base_url": null}}

Resolution (see :func:`limbo.llm.catalog.resolve_api_key`):
``[providers.<id>] api_key`` → auth.json → ``[llm] api_key`` → env var.
A ``[providers.<id>]`` key in config.toml stays the most specific override,
while a stored login beats the global ``[llm]`` key so logging in to a
second provider never sends it the first provider's key.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

DEFAULT_AUTH_PATH = Path.home() / ".limbo" / "auth.json"


@dataclass(frozen=True)
class Credential:
    key: str
    base_url: str | None = None


def auth_path() -> Path:
    override = os.environ.get("LIMBO_AUTH_FILE")
    return Path(override) if override else DEFAULT_AUTH_PATH


def _read(path: Path) -> dict[str, dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        warnings.warn(f"Could not read {path}: {e}; ignoring stored logins.", stacklevel=3)
        return {}
    return data if isinstance(data, dict) else {}


def load_all(path: Path | None = None) -> dict[str, Credential]:
    out: dict[str, Credential] = {}
    for provider, entry in _read(path or auth_path()).items():
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            base_url = entry.get("base_url")
            out[provider] = Credential(key, base_url if isinstance(base_url, str) else None)
    return out


def get(provider: str, path: Path | None = None) -> Credential | None:
    return load_all(path).get(provider)


def _write(data: dict[str, dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def save(provider: str, key: str, base_url: str | None = None, path: Path | None = None) -> None:
    path = path or auth_path()
    data = _read(path)
    entry: dict[str, object] = {"type": "api", "key": key.strip()}
    if base_url:
        entry["base_url"] = base_url.strip()
    data[provider] = entry
    _write(data, path)


def remove(provider: str, path: Path | None = None) -> bool:
    path = path or auth_path()
    data = _read(path)
    if provider not in data:
        return False
    del data[provider]
    _write(data, path)
    return True


def mask(key: str) -> str:
    """``sk-abc…wxyz`` — enough to tell keys apart, never enough to use one."""
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:5]}…{key[-4:]}"
