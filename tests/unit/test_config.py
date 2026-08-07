"""Unit tests for core configuration."""

from pathlib import Path

import pytest

from pioneer.core.config import Settings, get_settings, reset_settings_cache
from pioneer.core.exceptions import ConfigurationError


@pytest.mark.unit
def test_settings_defaults() -> None:
    settings = Settings(env="test")
    assert settings.default_model == "gpt-4o-mini"
    assert settings.seed == 42


@pytest.mark.unit
def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIONEER_DEFAULT_MODEL", "gpt-4o")
    reset_settings_cache()
    settings = get_settings()
    assert settings.default_model == "gpt-4o"


@pytest.mark.unit
def test_settings_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_yaml(tmp_path / "nope.yaml")
