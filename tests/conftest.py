"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from pioneer.core.config import Settings, reset_settings_cache


@pytest.fixture(autouse=True)
def test_env() -> Generator[None, None, None]:
    """Isolate tests with a clean settings environment."""
    os.environ["PIONEER_ENV"] = "test"
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings(env="test")
