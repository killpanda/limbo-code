"""Configuration loading for Limbo."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import toml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from toml import TomlDecodeError

DEFAULT_CONFIG_PATH = Path.home() / ".limbo" / "config.toml"

# Single source of truth for safety defaults; tools fall back to these when
# constructed without explicit values.
DEFAULT_DANGEROUS_COMMANDS = ["rm", "git reset --hard"]
DEFAULT_SENSITIVE_FILES = [".env", "id_rsa", "id_ed25519", ".ssh"]


class LLMConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
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


class UIConfig(BaseModel):
    # Textual built-in theme name, e.g. "textual-dark", "dracula", "nord".
    theme: str | None = None


class SafetyConfig(BaseModel):
    dangerous_commands: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DANGEROUS_COMMANDS)
    )
    sensitive_files: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_FILES)
    )


class ToolsConfig(BaseModel):
    bash_enabled: bool = True
    # Execute multiple tool calls from one assistant turn concurrently.
    # Set to false to fall back to strict sequential execution.
    parallel: bool = True


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


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)


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
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        warnings.warn(
            f"Invalid config file {path}: {e}. Using defaults.",
            stacklevel=2,
        )
        return Config()
