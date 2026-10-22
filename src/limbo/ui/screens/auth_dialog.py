"""Paste-a-key dialog, opened from the model picker for a provider with no key.

Checks the key against the provider (``limbo.llm.verify``) before saving it
to ``~/.limbo/auth.json``. A definite rejection keeps the dialog open; an
unreachable endpoint saves anyway and says it could not verify. The
``custom`` variant also asks for an endpoint and a model id so any
OpenAI-compatible server works without touching config.toml.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Input, Static

from limbo import auth
from limbo.llm import verify
from limbo.llm.catalog import GENERIC_OPENAI, ProviderSpec, provider_display_name
from limbo.ui.screens.dialog import Dialog


@dataclass(frozen=True)
class AuthResult:
    provider: str
    model: str | None = None  # set by the custom-endpoint form


class AuthDialog(Dialog[AuthResult | None]):
    HINTS = (("tab", "下一项"), ("enter", "验证并保存"), ("esc", "取消"))

    def __init__(self, provider: ProviderSpec, env_hint: str | None = None) -> None:
        super().__init__()
        self._provider = provider
        self._env_hint = env_hint
        self._busy = False
        self.TITLE_TEXT = f"登录 {provider_display_name(provider.id)}"

    @property
    def custom(self) -> bool:
        return self._provider.id == GENERIC_OPENAI.id

    def compose_body(self) -> ComposeResult:
        yield Static(classes="auth-intro", markup=False)
        if self.custom:
            yield Input(placeholder="https://your-endpoint/v1", id="auth-base-url")
            yield Input(placeholder="模型 id，例如 qwen3-coder", id="auth-model")
        yield Input(placeholder="粘贴 API key", password=True, id="auth-key")
        if not self.custom:
            yield Input(
                placeholder=f"endpoint（可选，默认 {self._provider.base_url}）",
                id="auth-base-url",
            )
        yield Static(classes="auth-status", markup=False)

    def on_mount(self) -> None:
        super().on_mount()
        v = self.app.current_theme.variables
        intro = Text("key 会保存在 ", Style(color=v.get("text-secondary", "")))
        intro.append("~/.limbo/auth.json", Style(bold=True))
        intro.append("（仅自己可读）", Style(color=v.get("text-secondary", "")))
        if self._env_hint:
            intro.append(f"\n也可以设置 ${self._env_hint}", Style(color=v.get("text-tertiary", "")))
        self.query_one(".auth-intro", Static).update(intro)
        first = "#auth-base-url" if self.custom else "#auth-key"
        self.query_one(first, Input).focus()

    def _status(self, message: str, tone: str = "text-secondary") -> None:
        theme = self.app.current_theme
        color = {
            "error": str(theme.error),
            "success": str(theme.success),
        }.get(tone, theme.variables.get(tone, ""))
        self.query_one(".auth-status", Static).update(Text(message, Style(color=color)))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if not self._busy:
            self.run_worker(self._submit(), exclusive=True)

    async def _submit(self) -> None:
        key = self.query_one("#auth-key", Input).value.strip()
        base_url = self.query_one("#auth-base-url", Input).value.strip() or None
        model = self.query_one("#auth-model", Input).value.strip() if self.custom else None
        if self.custom and not (base_url and model):
            self._status("自定义端点需要 endpoint 和模型 id", "error")
            return
        if not key:
            self._status("先粘贴一个 key", "error")
            self.query_one("#auth-key", Input).focus()
            return
        self._busy = True
        self._status("验证中…")
        try:
            result = await verify.check_key(self._provider, key, base_url)
        finally:
            self._busy = False
        if result.status == verify.INVALID:
            self._status(f"✗ {result.detail}", "error")
            return
        auth.save(self._provider.id, key, base_url)
        self._status(f"✓ {result.detail} · 已保存", "success")
        self.dismiss(AuthResult(self._provider.id, model))
