"""Unit tests for scenario runner and advisory."""

import pytest

from pioneer.intelligence.advisory.advisor import BusinessAdvisor
from pioneer.intelligence.ingestion.feeds import FeedItem, FeedCategory
from pioneer.intelligence.samples import sample_portfolio
from pioneer.intelligence.scenarios.runner import AutonomousScenarioRunner
from datetime import UTC, datetime


@pytest.mark.unit
def test_autonomous_scenario_suite() -> None:
    portfolio = sample_portfolio()
    result = AutonomousScenarioRunner().run_autonomous_suite(portfolio, count=4)
    assert len(result.outcomes) == 4
    assert result.highest_risk_scenario is not None
    assert result.baseline.overall_score >= 0.0


@pytest.mark.unit
def test_scenario_increases_risk_under_stress() -> None:
    portfolio = sample_portfolio()
    runner = AutonomousScenarioRunner()
    result = runner.run_autonomous_suite(portfolio)
    assert any(outcome.delta >= 0.0 for outcome in result.outcomes)


@pytest.mark.unit
def test_business_advisor_generates_recommendations() -> None:
    portfolio = sample_portfolio()
    report = BusinessAdvisor().generate_report(portfolio, run_scenarios=True)
    assert report.overall_risk_score > 0.0
    assert len(report.recommendations) >= 2
    assert report.executive_summary


@pytest.mark.unit
def test_advisor_includes_news_signals() -> None:
    portfolio = sample_portfolio()
    feed = [
        FeedItem(
            id="1",
            title="Major pipe burst downtown",
            summary="Water main failure affects buildings",
            url="https://example.com",
            source="test",
            category=FeedCategory.NEWS,
            published_at=datetime.now(UTC),
            keywords=["pipe", "burst"],
            relevance_score=0.8,
        )
    ]
    report = BusinessAdvisor().generate_report(portfolio, feed_items=feed)
    assert len(report.news_signals) == 1
