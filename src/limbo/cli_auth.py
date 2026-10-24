"""``limbo auth …`` / ``limbo models`` subcommands and first-run login.

Plain-terminal counterparts of the TUI model dialog: ``auth login`` asks
for a provider and a key (masked), checks it against the provider and
stores it in ``~/.limbo/auth.json``; ``auth list`` shows where every
provider's key comes from; ``models`` lists the catalog with availability.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from typing import TextIO

from limbo import auth
from limbo.config import Config, load_config
from limbo.llm.catalog import (
    CATALOG,
    MAGPIE,
    MAGPIE_PREFIX,
    ModelSpec,
    ProviderSpec,
    api_key_source,
    provider_display_name,
    providers,
    resolve_api_key,
    resolve_api_key_env,
    resolve_base_url,
    resolve_model,
    thinking_choices,
)
from limbo.llm.verify import INVALID, check_key, list_model_ids


def _first_model(provider: ProviderSpec) -> ModelSpec:
    return next(spec for spec in CATALOG.values() if spec.provider.id == provider.id)


def _context(tokens: int) -> str:
    return f"{tokens / 1_000_000:g}M" if tokens >= 1_000_000 else f"{tokens // 1024}K"


def pick_provider(stdin: TextIO, stdout: TextIO) -> ProviderSpec | None:
    options = providers()
    print("选择提供商：", file=stdout)
    for n, provider in enumerate(options, 1):
        print(f"  {n}. {provider_display_name(provider.id)}  ({provider.id})", file=stdout)
    stdout.write("编号或 id：")
    stdout.flush()
    answer = stdin.readline().strip()
    for n, provider in enumerate(options, 1):
        if answer in (str(n), provider.id):
            return provider
    print(f"没有这个提供商：{answer!r}", file=stdout)
    return None


def login(
    provider_id: str | None,
    key: str | None = None,
    base_url: str | None = None,
    verify: bool = True,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if provider_id is None:
        provider = pick_provider(stdin, stdout)
        if provider is None:
            return 1
    else:
        found = [p for p in providers() if p.id == provider_id]
        if not found:
            known = ", ".join(p.id for p in providers())
            print(f"未知提供商 {provider_id!r}（可选：{known}）", file=sys.stderr)
            return 1
        provider = found[0]
    name = provider_display_name(provider.id)
    if key is None:
        default = provider.default_api_key
        prompt = f"{name} API key（输入不回显{f'，回车用默认 {default}' if default else ''}）："
        if stdin.isatty():
            key = getpass.getpass(prompt)
        else:
            key = stdin.readline()
        key = key.strip() or (default or "")
    key = key.strip()
    if not key:
        print("没有输入 key，已取消", file=sys.stderr)
        return 1
    if verify:
        print("验证中…", file=stdout)
        result = asyncio.run(check_key(provider, key, base_url))
        print(f"{'✓' if result.ok else '✗'} {result.detail}", file=stdout)
        if result.status == INVALID:
            return 1
    auth.save(provider.id, key, base_url)
    print(f"已保存 {name} 的 key {auth.mask(key)} → {auth.auth_path()}", file=stdout)
    return 0


def list_logins(config: Config, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    stored = auth.load_all()
    for provider in providers():
        source = api_key_source(_first_model(provider), config)
        env = resolve_api_key_env(_first_model(provider), config)
        mark = "✓" if source else "·"
        line = f"{mark} {provider_display_name(provider.id):<18} {provider.id:<12}"
        if source == "auth.json":
            line += f" auth.json  {auth.mask(stored[provider.id].key)}"
        elif source:
            line += f" {source}"
        else:
            hint = f"limbo auth login {provider.id}" + (f" 或 ${env}" if env else "")
            line += f" 未配置（{hint}）"
        print(line, file=stdout)
    return 0


def logout(provider_id: str, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    if auth.remove(provider_id):
        print(f"已删除 {provider_display_name(provider_id)} 的已存 key", file=stdout)
        return 0
    print(f"auth.json 里没有 {provider_id}", file=stdout)
    return 1


def _magpie_specs(config: Config) -> list[ModelSpec]:
    """Models the running Magpie gateway serves, else the built-in examples."""
    seeded = [s for s in CATALOG.values() if s.provider is MAGPIE]
    key = resolve_api_key(seeded[0], config) or ""
    live = asyncio.run(
        list_model_ids(MAGPIE, key, resolve_base_url(seeded[0], config))
    )
    return [resolve_model(MAGPIE_PREFIX + m) for m in live] or seeded


def list_models(config: Config, query: str | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    current = config.llm.model
    for provider in providers():
        if provider is MAGPIE:
            specs = _magpie_specs(config)
        else:
            specs = [s for s in CATALOG.values() if s.provider.id == provider.id]
        if query:
            specs = [s for s in specs if query.lower() in s.id.lower()]
        if not specs:
            continue
        ready = api_key_source(specs[0], config) is not None
        header = provider_display_name(provider.id)
        print(f"{header}{'' if ready else '  （未配置 key）'}", file=stdout)
        for spec in specs:
            levels = [c for c in thinking_choices(spec) if c]
            parts = [_context(spec.context_window)]
            if levels:
                parts.append("思考 " + "/".join(levels))
            if spec.vision:
                parts.append("图像")
            mark = "●" if spec.id == current else " "
            print(f"  {mark} {spec.id:<24} {' · '.join(parts)}", file=stdout)
    return 0


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="limbo")
    sub = parser.add_subparsers(dest="command", required=True)
    auth_parser = sub.add_parser("auth", help="管理已存的 API key")
    auth_sub = auth_parser.add_subparsers(dest="action", required=True)
    login_parser = auth_sub.add_parser("login", help="保存一个提供商的 key")
    login_parser.add_argument("provider", nargs="?")
    login_parser.add_argument("--key", help="直接给出 key（默认交互输入）")
    login_parser.add_argument("--base-url", help="自定义 endpoint（中转/私有部署）")
    login_parser.add_argument("--no-verify", action="store_true", help="跳过联网验证")
    auth_sub.add_parser("list", help="列出各提供商 key 的来源")
    logout_parser = auth_sub.add_parser("logout", help="删除已存的 key")
    logout_parser.add_argument("provider")
    models_parser = sub.add_parser("models", help="列出可用模型")
    models_parser.add_argument("query", nargs="?")
    args = parser.parse_args(argv)

    if args.command == "models":
        return list_models(load_config(), args.query)
    if args.action == "login":
        return login(args.provider, args.key, args.base_url, verify=not args.no_verify)
    if args.action == "list":
        return list_logins(load_config())
    return logout(args.provider)


THINKING_WORDS = frozenset({"off", "on", "minimal", "low", "medium", "high", "xhigh", "max"})


def split_model_arg(value: str) -> tuple[str, str | None]:
    """``kimi-k3:high`` → (``kimi-k3``, ``high``); a bare id has no level.

    Only a known thinking word after the last ``:`` is split off, so model
    ids that contain colons themselves (``qwen3:32b``) pass through intact.
    """
    model, sep, level = value.rpartition(":")
    if sep and model and level.lower() in THINKING_WORDS and value not in CATALOG:
        return model, level.lower()
    return value, None
