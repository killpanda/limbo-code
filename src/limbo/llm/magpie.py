"""Live model list from the local Magpie gateway.

Magpie asks each upstream vendor which models it serves, so its list lives
on the user's machine and changes over time. The gateway's ``/v1/models``
carries each model's context window, output cap, modalities, reasoning
levels and native endpoints; reading it fills ``catalog.MAGPIE_LIVE`` so
``resolve_model("magpie/…")`` uses those values for the rest of the run.
``limbo models``, the model picker and startup with a Magpie model read it;
the catalog's example entry is only the fallback while the gateway is down.
"""

from __future__ import annotations

import asyncio

from limbo.config import Config
from limbo.llm.catalog import (
    CATALOG,
    MAGPIE,
    MAGPIE_EXAMPLE,
    MAGPIE_LIVE,
    MAGPIE_PREFIX,
    ModelSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from limbo.llm.verify import list_model_entries


def seeded_models() -> list[ModelSpec]:
    return [spec for spec in CATALOG.values() if spec.provider.id == MAGPIE.id]


async def live_models(config: Config) -> list[ModelSpec]:
    """Models the running gateway serves; empty when it is unreachable."""
    probe = CATALOG[MAGPIE_EXAMPLE]  # static entry: always the /v1 chat base
    entries = await list_model_entries(
        MAGPIE, resolve_api_key(probe, config) or "", resolve_base_url(probe, config)
    )
    for entry in entries:
        MAGPIE_LIVE[entry["id"]] = entry
    return [resolve_model(MAGPIE_PREFIX + entry["id"]) for entry in entries]


def load_for_startup(model_id: str, config: Config) -> None:
    """Read the gateway's list before a ``magpie/…`` model is first resolved,
    so compaction and the client see its real window and dialect."""
    if model_id.startswith(MAGPIE_PREFIX) and not MAGPIE_LIVE:
        asyncio.run(live_models(config))
