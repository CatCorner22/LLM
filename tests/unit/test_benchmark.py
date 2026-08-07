"""Unit tests for competitor benchmarking."""

import pytest

from pioneer.intelligence.benchmark.runner import BenchmarkRunner
from pioneer.intelligence.samples import sample_portfolio


@pytest.mark.unit
def test_benchmark_pioneer_wins_metrics() -> None:
    report = BenchmarkRunner().run(sample_portfolio())
    wins = sum(1 for m in report.metrics if m.pioneer_wins)
    assert wins >= 2
    assert report.summary
    assert len(report.disruptive_advantages) >= 3


@pytest.mark.unit
def test_benchmark_includes_competitors() -> None:
    report = BenchmarkRunner().run(sample_portfolio())
    assert len(report.competitor_results) == 3
    names = {c.name for c in report.competitor_results}
    assert "HeuristicRules v1" in names
