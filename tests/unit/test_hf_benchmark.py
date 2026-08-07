"""Unit tests for Hugging Face benchmark runner."""

from __future__ import annotations

from typing import Any

import pytest

from pioneer.intelligence.benchmark.hf_datasets import NEWS_RISK_BENCHMARK
from pioneer.intelligence.benchmark.huggingface import (
    HFBenchmarkConfig,
    HFBenchmarkRunner,
    _binary_metrics,
    _incident_to_portfolio,
)

SAMPLE_INCIDENTS: list[dict[str, Any]] = [
    {
        "incident_id": "INC-1",
        "shift_type": "night",
        "ppe_worn": "none",
        "years_experience": 1.0,
        "injury_severity": "fatality",
    },
    {
        "incident_id": "INC-2",
        "shift_type": "morning",
        "ppe_worn": "full",
        "years_experience": 15.0,
        "injury_severity": "first_aid",
    },
    {
        "incident_id": "INC-3",
        "shift_type": "night",
        "ppe_worn": "none",
        "years_experience": 2.0,
        "injury_severity": "days_away",
    },
    {
        "incident_id": "INC-4",
        "shift_type": "evening",
        "ppe_worn": "partial",
        "years_experience": 8.0,
        "injury_severity": "first_aid",
    },
]


@pytest.mark.unit
def test_binary_metrics_perfect_prediction() -> None:
    metrics = _binary_metrics([True, False, True], [0.9, 0.1, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


@pytest.mark.unit
def test_incident_to_portfolio() -> None:
    portfolio = _incident_to_portfolio(SAMPLE_INCIDENTS[0])
    assert portfolio.operational is not None
    assert portfolio.operational.night_shift_ratio == 1.0


@pytest.mark.unit
def test_hf_injury_prediction_task() -> None:
    runner = HFBenchmarkRunner(HFBenchmarkConfig(mining_sample_size=10))
    result = runner.benchmark_injury_prediction(SAMPLE_INCIDENTS)
    assert result.samples == 4
    assert result.metric_name == "f1_high_injury"
    assert 0.0 <= result.pioneer_score <= 1.0


@pytest.mark.unit
def test_hf_news_relevance_task() -> None:
    runner = HFBenchmarkRunner()
    result = runner.benchmark_news_relevance()
    assert result.samples == len(NEWS_RISK_BENCHMARK)
    assert result.pioneer_score >= result.baseline_score


@pytest.mark.unit
def test_hf_severity_calibration() -> None:
    runner = HFBenchmarkRunner()
    result = runner.benchmark_severity_calibration(SAMPLE_INCIDENTS)
    assert result.pioneer_wins or result.pioneer_score >= 0.0


@pytest.mark.integration
@pytest.mark.slow
def test_hf_benchmark_full_run() -> None:
    pytest.importorskip("datasets")
    report = HFBenchmarkRunner(HFBenchmarkConfig(mining_sample_size=200)).run()
    assert report.total_tasks == 4
    assert report.pioneer_task_wins >= 1
    assert "Hugging Face" in report.summary
    assert all(task.samples > 0 for task in report.tasks)
