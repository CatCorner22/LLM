"""Unit tests for evaluation metrics."""

import pytest

from pioneer.evaluation.metrics import MetricRegistry, exact_match, f1_score, pass_at_k


@pytest.mark.unit
def test_exact_match() -> None:
    assert exact_match(["Hello"], ["hello"]) == 1.0
    assert exact_match(["foo"], ["bar"]) == 0.0


@pytest.mark.unit
def test_f1_score() -> None:
    score = f1_score(["the cat sat"], ["the cat"])
    assert 0.0 < score <= 1.0


@pytest.mark.unit
def test_pass_at_k() -> None:
    assert pass_at_k(num_samples=10, num_correct=3, k=1) == pytest.approx(0.3)


@pytest.mark.unit
def test_metric_registry() -> None:
    score = MetricRegistry.compute("exact_match", ["a"], ["a"])
    assert score == 1.0
    assert "exact_match" in MetricRegistry.available()
