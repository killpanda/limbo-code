"""Model picker modal: list catalog models grouped by provider."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from limbo.config import Config
from limbo.llm.catalog import (
    CATALOG,
    ModelSpec,
    resolve_api_key,
    resolve_api_key_env,
)


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


class ModelPicker(Screen[str | None]):
    """Modal model list grouped by provider; Enter switches, Esc cancels.

    Providers without a resolvable API key are dimmed with a hint; selecting
    one of their models keeps the picker open and explains instead of
    switching into an auth error. Dismisses with the selected model id, or
    None when cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    def __init__(self, config: Config, current_model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._current = current_model
        # (model_id | None for provider headers, provider_available) aligned
        # with the ListView items, in display order.
        self._entries: list[tuple[str | None, bool]] = []

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []
        for provider_id, specs in provider_groups():
            available = resolve_api_key(specs[0], self._config) is not None
            header = f"▸ {provider_id}"
            if not available:
                env = resolve_api_key_env(specs[0], self._config)
                hint = f"${env}" if env else "config"
                header += f"（未配置 API key: {hint}）"
            items.append(
                ListItem(
                    Label(header, markup=False),
                    classes="provider-header",
                    disabled=True,
                )
            )
            self._entries.append((None, available))
            for spec in specs:
                items.append(
                    ListItem(
                        Label(self._format_model(spec, available), markup=False),
                        classes="" if available else "unavailable",
                    )
                )
                self._entries.append((spec.id, available))
        with Vertical(id="model-picker"):
            yield Static(
                "选择模型 · Enter 切换 · Esc 取消",
                id="picker-title",
                markup=False,
            )
            yield ListView(*items)
            yield Static("", id="picker-hint", markup=False)

    def _format_model(self, spec: ModelSpec, available: bool) -> str:
        parts = [spec.id, _format_context(spec.context_window)]
        if spec.reasoning:
            parts.append("reasoning")
        if spec.id == self._current:
            parts.append("当前")
        return "    " + " · ".join(parts)

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is None or not (0 <= index < len(self._entries)):
            return
        model_id, available = self._entries[index]
        if model_id is None:
            return  # provider header row
        if not available:
            spec = CATALOG[model_id]
            env = resolve_api_key_env(spec, self._config)
            hint = f"${env}" if env else "[providers] / [llm] api_key"
            self.query_one("#picker-hint", Static).update(
                f"未配置 {spec.provider.id} 的 API key（{hint}），无法切换"
            )
            return
        self.dismiss(model_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
