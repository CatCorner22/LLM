"""Environment-aware configuration with validation and secrets handling."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pioneer.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional YAML."""

    model_config = SettingsConfigDict(
        env_prefix="PIONEER_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    env: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False

    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    artifacts_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    config_dir: Path = Field(default_factory=lambda: Path("configs"))

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # LLM providers
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    default_model: str = "gpt-4o-mini"
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Training
    seed: int = 42
    num_workers: int = Field(default=4, ge=0)
    checkpoint_dir: Path = Field(default_factory=lambda: Path("artifacts/checkpoints"))

    # Inference / serving
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    # Observability
    enable_telemetry: bool = False
    otel_endpoint: str | None = None
    metrics_port: int = Field(default=9090, ge=1, le=65535)

    @field_validator("data_dir", "artifacts_dir", "config_dir", "checkpoint_dir", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        for directory in (self.data_dir, self.artifacts_dir, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path, *, overrides: dict[str, object] | None = None) -> Settings:
        """Load settings from a YAML file with optional overrides."""
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if overrides:
            raw.update(overrides)

        return cls(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    settings = Settings()
    if settings.env != "test":
        settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """Clear settings cache — useful in tests."""
    get_settings.cache_clear()
