"""Cheap API-key check: list models on the provider's endpoint.

Used by ``limbo auth login`` and the TUI auth dialog so a pasted key is
confirmed before it is saved. Only a definite 401/403 counts as a bad key;
timeouts, relays without ``/models`` and other errors come back as
"unverified" so an offline login still works.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from limbo.llm.catalog import API_ANTHROPIC_MESSAGES, ProviderSpec

VALID, INVALID, UNVERIFIED = "valid", "invalid", "unverified"


@dataclass(frozen=True)
class KeyCheck:
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != INVALID


def _request(provider: ProviderSpec, key: str, base_url: str) -> tuple[str, dict[str, str]]:
    base = base_url.rstrip("/")
    headers = dict(provider.headers)
    if provider.api == API_ANTHROPIC_MESSAGES:
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    else:
        headers["Authorization"] = f"Bearer {key}"
        url = base + "/models"
    return url, headers


async def check_key(
    provider: ProviderSpec, key: str, base_url: str | None = None, timeout: float = 8.0
) -> KeyCheck:
    base = base_url or provider.base_url
    if not base:
        return KeyCheck(UNVERIFIED, "没有 base_url，跳过验证")
    url, headers = _request(provider, key, base)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        return KeyCheck(UNVERIFIED, f"无法连接：{type(e).__name__}")
    if response.status_code in (401, 403):
        return KeyCheck(INVALID, f"服务端拒绝了这个 key（HTTP {response.status_code}）")
    if response.is_success:
        return KeyCheck(VALID, "key 有效")
    return KeyCheck(UNVERIFIED, f"HTTP {response.status_code}，未能确认")
