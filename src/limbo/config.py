"""Configuration loading for Limbo."""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import toml  # type: ignore[import-untyped]
import tomlkit
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from toml import TomlDecodeError

from limbo.llm.catalog import known_provider_ids

DEFAULT_CONFIG_PATH = Path.home() / ".limbo" / "config.toml"


class LLMConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.2
    max_iterations: int = 50
    # Thinking control for reasoning models (e.g. kimi-k3: low|high|max,
    # deepseek-format Kimi models: on value or "off"). None = provider default.
    thinking_effort: str | None = None
    # Per-request output token cap; None = use the model catalog default.
    max_tokens: int | None = None
    # LLM request retry/timeout knobs (see limbo.llm.retry).
    max_retries: int = 3
    retry_base_delay: float = 1.0
    timeout: float = 600.0
    connect_timeout: float = 30.0

    @field_validator("max_iterations")
    @classmethod
    def _max_iterations_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_iterations must be at least 1")
        return value

    # The retry/timeout fields below clamp-and-warn instead of raising: a
    # single invalid value must not discard the whole config (load_config
    # falls back to *all* defaults on ValidationError).

    @field_validator("max_retries")
    @classmethod
    def _max_retries_clamped(cls, value: int) -> int:
        if value < 0:
            warnings.warn(
                f"max_retries={value} is invalid; clamped to 0 (retries disabled).",
                stacklevel=2,
            )
            return 0
        return value

    @field_validator("retry_base_delay")
    @classmethod
    def _retry_base_delay_clamped(cls, value: float) -> float:
        if value <= 0:
            warnings.warn(
                f"retry_base_delay={value} is invalid; reset to 1.0.",
                stacklevel=2,
            )
            return 1.0
        return value

    @field_validator("timeout")
    @classmethod
    def _timeout_clamped(cls, value: float) -> float:
        if value <= 0:
            warnings.warn(
                f"timeout={value} is invalid; reset to 600.0.",
                stacklevel=2,
            )
            return 600.0
        return value

    @field_validator("connect_timeout")
    @classmethod
    def _connect_timeout_clamped(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            warnings.warn(
                f"connect_timeout={value} is invalid; reset to 30.0.",
                stacklevel=2,
            )
            return 30.0
        timeout = info.data.get("timeout")
        if isinstance(timeout, (int, float)) and value > timeout:
            warnings.warn(
                f"connect_timeout={value} exceeds timeout={timeout}; "
                f"clamped to {timeout}.",
                stacklevel=2,
            )
            return float(timeout)
        return value

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_tokens must be at least 1")
        return value


class ProviderOverride(BaseModel):
    """[providers.<id>] section: per-provider overrides.

    Every field is optional; an unset field falls back to the catalog's
    built-in value for that provider (see limbo.llm.catalog). Overrides take
    precedence over the global [llm] settings — see resolve_base_url /
    resolve_api_key for the full resolution order.
    """

    base_url: str | None = None
    api_key: str | None = None
    # Rename the credential env var (e.g. CODEX_API_KEY instead of the
    # catalog default). Ignored when api_key is set directly.
    api_key_env: str | None = None
    # Extra HTTP headers sent on every request to this provider (merged over
    # the provider's built-in headers, e.g. relay-specific auth headers).
    headers: dict[str, str] = Field(default_factory=dict)


class KittyKeyboardMode(str, Enum):
    """How Limbo negotiates Textual's kitty keyboard protocol.

    Textual 8.2.8 requests the kitty keyboard protocol with ``REPORT_ALL_KEYS``.
    Inside the herdr multiplexer, herdr mirrors that request to the host
    terminal (e.g. Ghostty), and Ghostty then encodes IME commits as the
    physical commit key (space) while dropping the composed text — typing
    Chinese turns into spaces (pasting is unaffected, it bypasses the key
    encoder).

    Turning the protocol off entirely is not an option either: herdr then
    encodes Shift+Enter for the pane as a bare ``\r`` (indistinguishable from
    Enter), so the input box can no longer insert a newline. Requesting only
    ``DISAMBIGUATE_ESCAPE_CODES`` is the middle ground: herdr leaves the host
    flags alone (IME text still arrives as raw UTF-8) and forwards Shift+Enter
    as ``ESC [ 13 ; 2 u``.
    """

    # Inside the herdr multiplexer (detected via HERDR_ENV) request only the
    # disambiguate flag; the full Textual request everywhere else. Safe
    # default for CJK users.
    AUTO = "auto"
    # Always request the full kitty keyboard protocol (Textual's default).
    ENABLED = "enabled"
    # Never request it — forces raw/legacy terminal input everywhere
    # (Shift+Enter is then indistinguishable from Enter inside herdr).
    DISABLED = "disabled"

    def disable_textual_kitty_key(self) -> bool:
        """Whether Textual's kitty keyboard protocol should be disabled."""
        return self is KittyKeyboardMode.DISABLED

    def disambiguate_only(self, *, in_herdr: bool) -> bool:
        """Whether to request only the kitty disambiguate flag."""
        return self is KittyKeyboardMode.AUTO and in_herdr


class UIConfig(BaseModel):
    # Theme name. Unset / "auto" follows the Omarchy system theme when it is
    # installed and uses "limbo-dusk" otherwise; "omarchy" always follows it.
    # Also accepts "limbo-dusk" / "limbo-dark" / "limbo-light" or any Textual
    # built-in theme (e.g. "textual-dark", "dracula", "nord").
    theme: str | None = None
    # Show the startup landing (gradient wordmark + tips) on fresh sessions.
    show_banner: bool = True
    # Animation level (Limbo Motion): "full" / "reduced" / "off". Unset =
    # full, or reduced inside an SSH session. LIMBO_MOTION overrides it.
    motion: str | None = None
    # Kitty keyboard protocol handling (see KittyKeyboardMode):
    #   auto     - enable normally; inside the herdr multiplexer request only
    #              the disambiguate flag, the full set breaks IME/Chinese
    #              input there (default)
    #   enabled  - always request the kitty keyboard protocol
    #   disabled - never request it (forces raw/legacy terminal input)
    kitty_keyboard: KittyKeyboardMode = KittyKeyboardMode.AUTO


class ToolsConfig(BaseModel):
    bash_enabled: bool = True
    # Execute multiple tool calls from one assistant turn concurrently.
    # Set to false to fall back to strict sequential execution.
    parallel: bool = True


class WebSearchSettings(BaseModel):
    """[web_search] section: the ``web_search`` tool (limbo.websearch)."""

    enabled: bool = True
    # Backend id from limbo.websearch.PROVIDERS (only "deepseek" for now).
    provider: str = "deepseek"
    # Key for the search provider. Unset: inherit the matching model
    # provider's key (DeepSeek: [providers.deepseek] / auth login /
    # $DEEPSEEK_API_KEY). With no key at all the tool is not offered.
    api_key: str | None = None
    # Endpoint override (None = the provider's default).
    base_url: str | None = None
    # Results per search when the model does not ask for a count.
    count: int = 10
    timeout: float = 30.0

    @field_validator("count")
    @classmethod
    def _count_clamped(cls, value: int) -> int:
        if value < 1:
            warnings.warn(f"web_search count={value} is invalid; reset to 10.", stacklevel=2)
            return 10
        return value

    @field_validator("timeout")
    @classmethod
    def _timeout_clamped(cls, value: float) -> float:
        if value <= 0:
            warnings.warn(f"web_search timeout={value} is invalid; reset to 30.0.", stacklevel=2)
            return 30.0
        return value


class DailySettings(BaseModel):
    """[daily] section: the daily (non-coding) session posture."""

    # Workdir of `limbo daily` sessions (notes, research output, downloads).
    workdir: str = "~/limbo-daily"
    # Model for `limbo daily` (empty = [llm] model).
    model: str = ""
    # Persistent memory (~/.limbo/memory/): the memory tool + prompt snapshot.
    memory: bool = True
    user_chars: int = 1500
    notes_chars: int = 2500
    # Remind the model to consider memory after this many user turns
    # without a memory write (0 = never).
    memory_nudge_turns: int = 10
    # Automatic skill review (/learn) after a clean turn once this many tool
    # rounds ran since the last skill update or review, and the session
    # showed a signal worth learning from (0 = only /learn). Reviews that
    # save nothing double it, up to 8×; see limbo/learn_trigger.py.
    learn_after_rounds: int = 10
    # At least this many turns between two automatic reviews.
    learn_cooldown_turns: int = 3
    # web_fetch: characters per page chunk, request timeout.
    fetch_max_chars: int = 20_000
    fetch_timeout: float = 30.0
    # TUI look while daily mode is on: "dawn" = warm variant of the current
    # theme + ☀ badges, "badge" = badges only (colours stay as they are).
    look: str = "dawn"

    @model_validator(mode="before")
    @classmethod
    def _renamed_learn_key(cls, data: Any) -> Any:
        # learn_after_tool_calls (single calls) became learn_after_rounds.
        if isinstance(data, dict) and "learn_after_tool_calls" in data:
            data = dict(data)
            old = data.pop("learn_after_tool_calls")
            data.setdefault("learn_after_rounds", old)
        return data

    @field_validator("look")
    @classmethod
    def _look_known(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ("dawn", "badge"):
            warnings.warn(f"daily look={value!r} is unknown; reset to 'dawn'.", stacklevel=2)
            return "dawn"
        return value

    @field_validator("user_chars", "notes_chars", "fetch_max_chars")
    @classmethod
    def _positive_size(cls, value: int, info: ValidationInfo) -> int:
        default = cls.model_fields[info.field_name or ""].default
        if value < 100:
            warnings.warn(
                f"daily {info.field_name}={value} is too small; reset to {default}.",
                stacklevel=2,
            )
            return int(default)
        return value


class CompactionSettings(BaseModel):
    """[compaction] section: auto context-window compaction (LIM-14)."""

    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000

    @field_validator("reserve_tokens")
    @classmethod
    def _reserve_tokens_clamped(cls, value: int) -> int:
        if value < 1024:
            warnings.warn(
                f"reserve_tokens={value} is too small; reset to 16384.",
                stacklevel=2,
            )
            return 16_384
        return value

    @field_validator("keep_recent_tokens")
    @classmethod
    def _keep_recent_tokens_clamped(cls, value: int) -> int:
        if value < 1000:
            warnings.warn(
                f"keep_recent_tokens={value} is too small; reset to 20000.",
                stacklevel=2,
            )
            return 20_000
        return value


class GoalSettings(BaseModel):
    """[goal] section: closed-loop /goal mode (LIM-40)."""

    # Max verify attempts before the loop stops with a wrap-up summary.
    max_rounds: int = 10
    verify_timeout_ms: int = 600_000


class QueueSettings(BaseModel):
    """[queue] section: the queued-message inbox (steer / follow-up)."""

    # What happens to queued messages when a turn fails (the LLM errored
    # after its retries): "continue" feeds them into the next turn as
    # usual; "keep" stops the task and leaves them queued for the next
    # submission; "withdraw" stops the task and takes them out of the
    # queue (the TUI puts their text back into the input box). With
    # "keep"/"withdraw" a persistently failing provider cannot burn
    # through the queue one message per failed turn.
    after_failed_run: Literal["continue", "keep", "withdraw"] = "continue"


class SubagentSettings(BaseModel):
    """[subagents] section: background subagents and the task board."""

    # false: no supervisor at all (no board, no /agents, no task tools).
    enabled: bool = True
    # off: no task tools for the model (the board stays readable); suggest:
    # the model's plan waits for the user's approval; on: it runs at once.
    auto: Literal["off", "suggest", "on"] = "suggest"
    max_parallel: int = 3
    # Live (unfinished) tasks at a time. Subagents never plan themselves
    # (depth 1): they can only propose_task, which the main agent approves.
    max_tasks: int = 12
    # Per-task budget of fresh tokens (total minus prompt-cache hits;
    # 0 = unlimited). Over it, the current tool batch completes and the
    # task ends before its next request, reporting what it changed.
    task_token_budget: int = 400_000
    # Model for profiles with ``model: cheap`` (empty = the main model).
    cheap_model: str = ""
    # Which notices wake an idle main agent: question, alert, wave_done
    # (results are batched until no subagent is running), result (each).
    wake_main_on: list[str] = Field(
        default_factory=lambda: ["question", "alert", "wave_done"]
    )
    on_dep_fail: Literal["notify", "cascade"] = "notify"
    # Tasks that were running when the process stopped: ask (paused, the
    # user resumes with /task resume all), auto (resume), pause (paused).
    on_resume: Literal["ask", "auto", "pause"] = "ask"
    default_isolation: Literal["shared", "worktree"] = "shared"
    # Characters of a finished task's summary delivered to the main agent.
    result_chars: int = 2000
    # Auto checkpoint every N tool calls of a subagent (0 = off).
    auto_checkpoint_every: int = 8


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    goal: GoalSettings = Field(default_factory=GoalSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    subagents: SubagentSettings = Field(default_factory=SubagentSettings)
    daily: DailySettings = Field(default_factory=DailySettings)
    # [providers.<id>] per-provider overrides, keyed by catalog provider id
    # (e.g. [providers.glm], [providers.codex]).
    providers: dict[str, ProviderOverride] = Field(default_factory=dict)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    try:
        data: dict[str, Any] = toml.load(path)
    except TomlDecodeError as e:
        warnings.warn(
            f"Malformed config file {path}: {e}. Using defaults.",
            stacklevel=2,
        )
        return Config()
    except OSError as e:
        warnings.warn(
            f"Could not read config file {path}: {e}. Using defaults.",
            stacklevel=2,
        )
        return Config()
    # [provider.<id>] (singular) is a common typo for [providers.<id>]: the
    # table lands under a top-level "provider" key nothing reads and pydantic
    # ignores it silently, so the configured key never resolves. Warn instead
    # of failing, pointing at the correct section name.
    if isinstance(data.get("provider"), dict):
        warnings.warn(
            "config has a [provider.*] section (singular); Limbo expects "
            "[providers.<id>] — the section is ignored.",
            stacklevel=2,
        )
    # A [providers.<id>] section whose id no model can resolve to (see
    # known_provider_ids) is dead config — usually a typo or a section for a
    # provider the catalog doesn't know. Warn so it surfaces before the
    # "missing API key" error does. Guard with isinstance: a [[providers]]
    # array-of-tables (or a scalar [providers] key) is valid TOML but not the
    # dict shape we expect — pydantic rejects it with a clean ValidationError
    # fallback, so we must not crash the loop first.
    providers = data.get("providers")
    if isinstance(providers, dict):
        for pid in providers:
            if pid not in known_provider_ids():
                warnings.warn(
                    f"[providers.{pid}] matches no known provider and is never "
                    "used; check the section name.",
                    stacklevel=2,
                )
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        warnings.warn(
            f"Invalid config file {path}: {e}. Using defaults.",
            stacklevel=2,
        )
        return Config()


def save_model_to_config(model: str, path: Path | None = None) -> bool:
    """Persist the selected model to ``config.toml``, preserving comments.

    Uses tomlkit for a comment-preserving read-modify-write; creates a
    minimal config file when none exists. Returns False on any I/O or parse
    failure (a malformed file is never clobbered) so the caller can degrade
    to a session-only switch.
    """
    path = path or DEFAULT_CONFIG_PATH
    try:
        if path.exists():
            doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()
        llm = doc.get("llm")
        if not isinstance(llm, dict):
            llm = tomlkit.table()
            doc["llm"] = llm
        llm["model"] = model
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    except Exception:  # noqa: BLE001 - any failure degrades to session-only
        return False
    return True


def save_thinking_to_config(level: str | None, path: Path | None = None) -> bool:
    """Persist ``[llm] thinking_effort`` (None removes it); same contract as
    :func:`save_model_to_config`."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        if path.exists():
            doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()
        llm = doc.get("llm")
        if not isinstance(llm, dict):
            llm = tomlkit.table()
            doc["llm"] = llm
        if level is None:
            llm.pop("thinking_effort", None)
        else:
            llm["thinking_effort"] = level
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    except Exception:  # noqa: BLE001 - any failure degrades to session-only
        return False
    return True
