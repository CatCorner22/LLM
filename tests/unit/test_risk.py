"""Unit tests for risk assessment models."""

import pytest

from pioneer.intelligence.risk.accident import AccidentLikelihoodModel
from pioneer.intelligence.risk.building import BuildingRiskAssessor
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.infrastructure import InfrastructureRiskAssessor
from pioneer.intelligence.risk.models import (
    BuildingProfile,
    PipeProfile,
    RiskSeverity,
    weibull_failure_probability,
)
from pioneer.intelligence.risk.weather import WeatherRiskAssessor
from pioneer.intelligence.samples import sample_portfolio


@pytest.mark.unit
def test_weibull_failure_increases_with_age() -> None:
    young = weibull_failure_probability(10, shape=2.8, scale=50.0)
    old = weibull_failure_probability(60, shape=2.8, scale=50.0)
    assert old > young


@pytest.mark.unit
def test_building_risk_old_building_scores_higher() -> None:
    assessor = BuildingRiskAssessor()
    old = assessor.assess(
        BuildingProfile(asset_id="old", year_built=1920, structural_condition=0.5)
    )
    new = assessor.assess(
        BuildingProfile(asset_id="new", year_built=2015, structural_condition=0.95)
    )
    assert old.overall_score > new.overall_score


@pytest.mark.unit
def test_pipe_age_risk_cast_iron() -> None:
    assessor = InfrastructureRiskAssessor()
    result = assessor.assess_pipe(
        PipeProfile(asset_id="pipe", install_year=1950, material="cast_iron")
    )
    assert result.overall_score > 0.3
    assert any(f.name == "pipe_age_failure" for f in result.factors)


@pytest.mark.unit
def test_accident_model_returns_probabilities() -> None:
    portfolio = sample_portfolio()
    result = AccidentLikelihoodModel().predict(portfolio)
    assert result.accident_probability is not None
    assert 0.0 < result.accident_probability < 1.0
    assert result.injury_probability is not None


@pytest.mark.unit
def test_composite_engine() -> None:
    portfolio = sample_portfolio()
    assessment = CompositeRiskEngine().assess(portfolio)
    assert assessment.overall_score > 0.0
    assert assessment.severity in RiskSeverity
    assert len(assessment.factors) >= 4


@pytest.mark.unit
def test_weather_assessor() -> None:
    from pioneer.intelligence.risk.models import WeatherProfile

    result = WeatherRiskAssessor().assess(
        WeatherProfile(
            asset_id="w1",
            region="Gulf Coast",
            flood_risk_index=0.8,
            storm_probability_7d=0.6,
        )
    )
    assert result.overall_score > 0.4
