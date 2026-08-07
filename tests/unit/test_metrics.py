"""Unit tests for evaluation metrics."""

import pytest

from pioneer.evaluation.metrics import (
    MetricRegistry,
    exact_match,
    f1_score,
    normalize_answer,
    pass_at_k,
    squad_exact_match,
    squad_f1,
)


@pytest.mark.unit
def test_exact_match() -> None:
    assert exact_match(["Hello"], ["hello"]) == 1.0
    assert exact_match(["foo"], ["bar"]) == 0.0


@pytest.mark.unit
def test_normalize_answer_squad() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("  a   DOG  ") == "dog"


@pytest.mark.unit
def test_squad_exact_match_ignores_articles_and_punctuation() -> None:
    # Differs only by article + punctuation + case: plain EM misses, SQuAD EM hits.
    assert exact_match(["The dog."], ["a dog"]) == 0.0
    assert squad_exact_match(["The dog."], ["a dog"]) == 1.0


@pytest.mark.unit
def test_squad_f1_registered() -> None:
    assert squad_f1(["the cat sat"], ["cat sat"]) == pytest.approx(1.0)
    assert MetricRegistry.compute("squad_em", ["The Cat!"], ["cat"]) == 1.0
    assert "squad_f1" in MetricRegistry.available()


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
