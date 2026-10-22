"""Model picker (ctrl+l / /model): search, recents, and log in inline.

Models are grouped by provider; recently used ones are listed first.
Typing filters by model id or provider. Picking a model whose provider has
no key opens :class:`AuthDialog` to paste one, then switches. The last row
connects any OpenAI-compatible endpoint. Dismisses with the model id, or
None when cancelled.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Label, ListItem, ListView, Static

from limbo.config import Config
from limbo.llm.catalog import (
    CATALOG,
    GENERIC_OPENAI,
    ModelSpec,
    api_key_source,
    provider_display_name,
    resolve_api_key_env,
)
from limbo.ui import recent_models
from limbo.ui.screens.auth_dialog import AuthDialog, AuthResult
from limbo.ui.screens.dialog import Dialog

CUSTOM_ROW = "＋ 连接自定义 OpenAI 兼容端点…"


def _format_context(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M"
    return f"{tokens // 1024}K"


def provider_groups() -> list[tuple[str, list[ModelSpec]]]:
    """Catalog models grouped by provider, in first-appearance order."""
    groups: dict[str, list[ModelSpec]] = {}
    order: list[str] = []
    for spec in CATALOG.values():
        if spec.provider.id not in groups:
            groups[spec.provider.id] = []
            order.append(spec.provider.id)
        groups[spec.provider.id].append(spec)
    return [(pid, groups[pid]) for pid in order]


class ModelPicker(Dialog[str | None]):
    TITLE_TEXT = "模型"
    HINTS = (("↑↓", "选择"), ("enter", "切换"), ("shift+tab", "思考档位"), ("esc", "取消"))
    BINDINGS = [
        Binding("escape", "cancel", "取消", show=False),
        Binding("down", "cursor(1)", show=False, priority=True),
        Binding("up", "cursor(-1)", show=False, priority=True),
    ]

    def __init__(self, config: Config, current_model: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._config = config
        self._current = current_model
        # (model_id | None for headers | CUSTOM_ROW, provider_available),
        # aligned with the ListView items, in display order.
        self._entries: list[tuple[str | None, bool]] = []
        self._pending: str | None = None

    def compose_body(self) -> ComposeResult:
        yield Input(placeholder="搜索模型或提供商…", id="model-search")
        yield ListView(*self._items(""))
        yield Static("", id="picker-hint", markup=False)

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one(Input).focus()
        self._select_first()

    def _available(self, spec: ModelSpec) -> bool:
        return api_key_source(spec, self._config) is not None

    def _items(self, query: str) -> list[ListItem]:
        query = query.strip().lower()
        self._entries = []
        items: list[ListItem] = []

        def matches(spec: ModelSpec) -> bool:
            haystack = f"{spec.id} {spec.provider.id} {provider_display_name(spec.provider.id)}"
            return not query or query in haystack.lower()

        recent = [CATALOG[m] for m in recent_models.load() if m in CATALOG and m != self._current]
        recent = [spec for spec in recent if matches(spec)]
        if recent:
            items.append(ListItem(Label("◷ 最近", markup=False), classes="recent-header",
                                  disabled=True))
            self._entries.append((None, True))
            for spec in recent:
                items.append(self._row(spec, self._available(spec)))

        for provider_id, specs in provider_groups():
            specs = [spec for spec in specs if matches(spec)]
            if not specs:
                continue
            available = self._available(specs[0])
            header = f"▸ {provider_id}"
            if not available:
                env = resolve_api_key_env(specs[0], self._config)
                hint = f"${env}" if env else "config"
                header += f"（未配置 API key: {hint}）"
            items.append(
                ListItem(Label(header, markup=False), classes="provider-header", disabled=True)
            )
            self._entries.append((None, available))
            for spec in specs:
                items.append(self._row(spec, available))

        if not query or query in "custom 自定义 endpoint":
            items.append(ListItem(Label(CUSTOM_ROW, markup=False), classes="custom-row"))
            self._entries.append((CUSTOM_ROW, True))
        return items

    def _row(self, spec: ModelSpec, available: bool) -> ListItem:
        self._entries.append((spec.id, available))
        return ListItem(
            Label(self._format_model(spec, available), markup=False),
            classes="" if available else "unavailable",
        )

    def _format_model(self, spec: ModelSpec, available: bool) -> str:
        parts = [spec.id, _format_context(spec.context_window)]
        if spec.reasoning:
            parts.append("reasoning")
        if spec.id == self._current:
            parts.append("当前")
        return "    " + " · ".join(parts)

    def _select_first(self) -> None:
        view = self.query_one(ListView)
        for index, (model_id, _) in enumerate(self._entries):
            if model_id == self._current:
                view.index = index
                return
        for index, (model_id, _) in enumerate(self._entries):
            if model_id is not None:
                view.index = index
                return

    async def on_input_changed(self, event: Input.Changed) -> None:
        view = self.query_one(ListView)
        await view.clear()
        await view.extend(self._items(event.value))
        self._select_first()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._choose(self.query_one(ListView).index)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._choose(event.list_view.index)

    def action_cursor(self, delta: int) -> None:
        view = self.query_one(ListView)
        index = view.index if view.index is not None else -delta
        while 0 <= index + delta < len(self._entries):
            index += delta
            if self._entries[index][0] is not None:
                view.index = index
                return

    def _choose(self, index: int | None) -> None:
        if index is None or not (0 <= index < len(self._entries)):
            return
        model_id, available = self._entries[index]
        if model_id is None:
            return  # header row
        if model_id == CUSTOM_ROW:
            self.app.push_screen(AuthDialog(GENERIC_OPENAI), self._on_auth)
            return
        if not available:
            spec = CATALOG[model_id]
            env = resolve_api_key_env(spec, self._config)
            hint = f"${env}" if env else "[providers] / [llm] api_key"
            self.query_one("#picker-hint", Static).update(
                f"未配置 {spec.provider.id} 的 API key（{hint}），粘贴一个即可切换"
            )
            self._pending = model_id
            self.app.push_screen(AuthDialog(spec.provider, env), self._on_auth)
            return
        self.dismiss(model_id)

    def _on_auth(self, result: AuthResult | None) -> None:
        if result is None:
            return
        self.dismiss(result.model or self._pending)
