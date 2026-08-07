"""Autonomous scenario testing engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from pydantic import BaseModel

from pioneer.core.logging import get_logger
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.models import AssetPortfolio, RiskAssessment
from pioneer.intelligence.scenarios.definitions import DEFAULT_SCENARIOS, ScenarioDefinition

logger = get_logger(__name__)


class ScenarioOutcome(BaseModel):
    scenario_id: str
    scenario_name: str
    baseline_score: float
    stressed_score: float
    delta: float
    baseline_severity: str
    stressed_severity: str
    accident_probability: float | None = None
    injury_probability: float | None = None
    business_impact_summary: str


@dataclass
class ScenarioRunResult:
    asset_id: str
    baseline: RiskAssessment
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    highest_risk_scenario: str | None = None


class AutonomousScenarioRunner:
    """Automatically stress-test portfolios across diverse business scenarios."""

    def __init__(self, engine: CompositeRiskEngine | None = None) -> None:
        self.engine = engine or CompositeRiskEngine()

    def _apply_scenario(
        self, portfolio: AssetPortfolio, scenario: ScenarioDefinition
    ) -> AssetPortfolio:
        stressed = copy.deepcopy(portfolio)

        for param in scenario.parameters:
            if param.name == "flood_risk_index" and stressed.weather:
                stressed.weather.flood_risk_index = min(
                    1.0, stressed.weather.flood_risk_index + param.delta
                )
            elif param.name == "storm_probability_7d" and stressed.weather:
                stressed.weather.storm_probability_7d = min(
                    1.0, stressed.weather.storm_probability_7d + param.delta
                )
            elif param.name == "heat_wave_days_forecast" and stressed.weather:
                stressed.weather.heat_wave_days_forecast += int(param.delta)
            elif param.name == "occupancy_multiplier" and stressed.building:
                stressed.building.occupancy = int(stressed.building.occupancy * param.delta)
            elif param.name == "maintenance_backlog_days" and stressed.operational:
                stressed.operational.maintenance_backlog_days += param.delta
            elif param.name == "pipe_failure_multiplier" and stressed.pipes:
                for pipe in stressed.pipes:
                    pipe.inspection_score = max(0.0, pipe.inspection_score - 0.2)
                    pipe.soil_corrosivity = min(1.0, pipe.soil_corrosivity + 0.15)

        return stressed

    def _impact_summary(self, scenario: ScenarioDefinition, delta: float) -> str:
        if delta >= 0.25:
            level = "Severe"
        elif delta >= 0.1:
            level = "Moderate"
        else:
            level = "Low"
        return (
            f"{level} impact under '{scenario.name}': risk score change {delta:+.2f}. "
            f"Scenario type: {scenario.scenario_type.value}."
        )

    def run(
        self,
        portfolio: AssetPortfolio,
        scenarios: list[ScenarioDefinition] | None = None,
        *,
        min_probability: float = 0.0,
    ) -> ScenarioRunResult:
        scenario_list = scenarios or DEFAULT_SCENARIOS
        baseline = self.engine.assess(portfolio)
        outcomes: list[ScenarioOutcome] = []

        for scenario in scenario_list:
            if scenario.probability < min_probability:
                continue
            stressed_portfolio = self._apply_scenario(portfolio, scenario)
            stressed = self.engine.assess(stressed_portfolio)
            delta = stressed.overall_score - baseline.overall_score

            outcomes.append(
                ScenarioOutcome(
                    scenario_id=scenario.id,
                    scenario_name=scenario.name,
                    baseline_score=baseline.overall_score,
                    stressed_score=stressed.overall_score,
                    delta=delta,
                    baseline_severity=baseline.severity.value,
                    stressed_severity=stressed.severity.value,
                    accident_probability=stressed.accident_probability,
                    injury_probability=stressed.injury_probability,
                    business_impact_summary=self._impact_summary(scenario, delta),
                )
            )

        outcomes.sort(key=lambda item: item.delta, reverse=True)
        highest = outcomes[0].scenario_name if outcomes else None

        logger.info(
            "scenario_run_complete",
            asset_id=baseline.asset_id,
            scenarios_tested=len(outcomes),
            highest_risk=highest,
        )

        return ScenarioRunResult(
            asset_id=baseline.asset_id,
            baseline=baseline,
            outcomes=outcomes,
            highest_risk_scenario=highest,
        )

    def run_autonomous_suite(self, portfolio: AssetPortfolio, count: int = 6) -> ScenarioRunResult:
        """Run the full default scenario battery sorted by likelihood."""
        ranked = sorted(DEFAULT_SCENARIOS, key=lambda s: s.probability, reverse=True)
        return self.run(portfolio, scenarios=ranked[:count])
