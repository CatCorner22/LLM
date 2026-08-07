"""Composite risk engine combining all assessors."""

from __future__ import annotations

from pioneer.intelligence.risk.accident import AccidentLikelihoodModel
from pioneer.intelligence.risk.building import BuildingRiskAssessor
from pioneer.intelligence.risk.governance import GovernanceRiskAssessor
from pioneer.intelligence.risk.infrastructure import InfrastructureRiskAssessor
from pioneer.intelligence.risk.models import (
    AssetPortfolio,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    severity_from_score,
)
from pioneer.intelligence.risk.weather import WeatherRiskAssessor


class CompositeRiskEngine:
    """Orchestrate building, weather, infrastructure, and accident risk models."""

    CATEGORY_WEIGHTS: dict[RiskCategory, float] = {
        RiskCategory.BUILDING: 0.22,
        RiskCategory.WEATHER: 0.18,
        RiskCategory.INFRASTRUCTURE: 0.22,
        RiskCategory.ACCIDENT: 0.28,
        RiskCategory.GOVERNANCE: 0.1,
    }

    def __init__(self) -> None:
        self.building = BuildingRiskAssessor()
        self.weather = WeatherRiskAssessor()
        self.infrastructure = InfrastructureRiskAssessor()
        self.accident = AccidentLikelihoodModel()
        self.governance = GovernanceRiskAssessor()

    def assess(self, portfolio: AssetPortfolio) -> RiskAssessment:
        partials: list[RiskAssessment] = []

        if portfolio.building:
            partials.append(self.building.assess(portfolio.building))
        if portfolio.weather:
            partials.append(self.weather.assess(portfolio.weather))
        pipe_assessment = self.infrastructure.assess_portfolio_pipes(portfolio.pipes)
        if pipe_assessment:
            partials.append(pipe_assessment)
        accident_assessment = self.accident.predict(portfolio)
        partials.append(accident_assessment)
        governance_assessment = self.governance.assess(portfolio)
        if governance_assessment:
            partials.append(governance_assessment)

        all_factors: list[RiskFactor] = []
        category_scores: dict[RiskCategory, list[float]] = {}

        for partial in partials:
            all_factors.extend(partial.factors)
            for factor in partial.factors:
                category_scores.setdefault(factor.category, []).append(factor.score)

        weighted_sum = 0.0
        weight_total = 0.0
        for category, weight in self.CATEGORY_WEIGHTS.items():
            scores = category_scores.get(category)
            if not scores:
                continue
            category_avg = sum(scores) / len(scores)
            weighted_sum += category_avg * weight
            weight_total += weight

        overall = weighted_sum / weight_total if weight_total else 0.0

        asset_id = partials[0].asset_id if partials else "portfolio"
        return RiskAssessment(
            asset_id=asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=all_factors,
            accident_probability=accident_assessment.accident_probability,
            injury_probability=accident_assessment.injury_probability,
            metadata={
                "assessor": "composite",
                "partial_assessments": str(len(partials)),
            },
        )
