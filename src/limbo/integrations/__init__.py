"""Integrations with external terminal multiplexers / agent orchestrators.

Each integration lives in its own module (e.g. ``herdr``) and implements
the ``IntegrationReporter`` protocol from ``base``. Its ``from_env()``
returns ``None`` when Limbo is not running inside that tool, so the
integration is a strict no-op elsewhere. ``create_reporters()`` collects
every active integration into one ``CompositeReporter`` — the only object
call sites ever touch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from limbo.integrations.base import (
    AgentState,
    CompositeReporter,
    IntegrationReporter,
)
from limbo.integrations.herdr import HerdrReporter

__all__ = [
    "AgentState",
    "CompositeReporter",
    "HerdrReporter",
    "IntegrationReporter",
    "create_reporters",
]


def create_reporters(env: Mapping[str, str] | None = None) -> CompositeReporter:
    """Detect active integrations from the environment.

    Register new integrations here: append ``XReporter.from_env(env)`` when
    it returns a reporter. Order is irrelevant; each reporter is
    independent and fire-and-forget.
    """
    env = os.environ if env is None else env
    reporters: list[IntegrationReporter] = []
    if (herdr := HerdrReporter.from_env(env)) is not None:
        reporters.append(herdr)
    return CompositeReporter(reporters)
