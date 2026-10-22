"""Recently used models (``~/.limbo/recent_models.json``), newest first."""

from __future__ import annotations

import json

import limbo.config as config_module

LIMIT = 5


def _path():  # noqa: ANN202 - Path; resolved late so tests can redirect it
    return config_module.DEFAULT_CONFIG_PATH.parent / "recent_models.json"


def load() -> list[str]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in data if isinstance(m, str)][:LIMIT] if isinstance(data, list) else []


def push(model_id: str) -> None:
    models = [model_id, *(m for m in load() if m != model_id)][:LIMIT]
    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(models), encoding="utf-8")
    except OSError:
        pass
