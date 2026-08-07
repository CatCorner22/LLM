"""Unit tests for registry."""

import pytest

from pioneer.core.exceptions import ValidationError
from pioneer.core.registry import Registry


@pytest.mark.unit
def test_registry_register_and_get() -> None:
    registry: Registry[str] = Registry("test")
    registry.register("alpha", "value")
    assert registry.get("alpha") == "value"


@pytest.mark.unit
def test_registry_duplicate_raises() -> None:
    registry: Registry[str] = Registry("test")
    registry.register("alpha", "one")
    with pytest.raises(ValidationError):
        registry.register("alpha", "two")


@pytest.mark.unit
def test_registry_unknown_raises() -> None:
    registry: Registry[str] = Registry("test")
    with pytest.raises(ValidationError):
        registry.get("missing")
