"""Configuration loading for Limbo."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import toml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError
from toml import TomlDecodeError

DEFAULT_CONFIG_PATH = Path.home() / ".limbo" / "config.toml"


class LLMConfig(BaseModel):
    provider: str = "openai"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_iterations: int = 10


class UIConfig(BaseModel):
    pass


class SafetyConfig(BaseModel):
    dangerous_commands: list[str] = Field(
        default_factory=lambda: ["rm", "git reset --hard"]
    )
    sensitive_files: list[str] = Field(
        default_factory=lambda: [".env", "id_rsa", "id_ed25519", ".ssh"]
    )


class ToolsConfig(BaseModel):
    bash_enabled: bool = True


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


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
