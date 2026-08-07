"""Unit tests for exception hierarchy."""

import pytest

from pioneer.core.exceptions import PioneerError, ValidationError


@pytest.mark.unit
def test_pioneer_error_to_dict() -> None:
    error = PioneerError("something failed", code="TEST", details={"key": "value"})
    payload = error.to_dict()
    assert payload["error"] == "TEST"
    assert payload["message"] == "something failed"
    assert payload["details"]["key"] == "value"


@pytest.mark.unit
def test_validation_error_inherits() -> None:
    error = ValidationError("invalid input")
    assert isinstance(error, PioneerError)
