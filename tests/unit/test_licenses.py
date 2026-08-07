"""Unit tests for open source license tooling."""

import pytest
from scripts.check_licenses import ALLOWLIST, _load_allowlist, _project_packages


@pytest.mark.unit
def test_allowlist_includes_mit_and_bsd() -> None:
    allowlist = _load_allowlist()
    assert "MIT" in allowlist
    assert "BSD" in allowlist
    assert "Apache" in allowlist


@pytest.mark.unit
def test_allowlist_file_exists() -> None:
    assert ALLOWLIST.exists()


@pytest.mark.unit
def test_project_packages_non_empty() -> None:
    pytest.importorskip("pipdeptree")
    packages = _project_packages()
    assert "pydantic" in packages
    assert "httpx" in packages
