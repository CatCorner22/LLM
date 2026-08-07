"""Competitor baseline implementations for benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pioneer.intelligence.risk.models import AssetPortfolio, RiskSeverity


@dataclass
class CompetitorResult:
    name: str
    overall_score: float
    severity: RiskSeverity
    accident_probability: float | None
    recommendation_count: int
    scenario_coverage: float
    latency_ms: float
    notes: str


class CompetitorBaseline(ABC):
    """Abstract competitor or legacy system for benchmark comparison."""

    name: str

    @abstractmethod
    def assess(self, portfolio: AssetPortfolio) -> CompetitorResult:
        """Produce a risk assessment using competitor methodology."""


class HeuristicRulesBaseline(CompetitorBaseline):
    """Simple rule-based risk tool (typical legacy competitor)."""

    name = "HeuristicRules v1"

    def assess(self, portfolio: AssetPortfolio) -> CompetitorResult:
        import time

        start = time.perf_counter()
        score = 0.2
        if portfolio.building:
            age = 2026 - portfolio.building.year_built
            if age > 50:
                score += 0.2
            if portfolio.building.structural_condition < 0.6:
                score += 0.15
        if portfolio.pipes:
            oldest = max(2026 - p.install_year for p in portfolio.pipes)
            if oldest > 40:
                score += 0.2
        if portfolio.weather and portfolio.weather.flood_risk_index > 0.5:
            score += 0.15
        score = min(1.0, score)
        latency = (time.perf_counter() - start) * 1000

        return CompetitorResult(
            name=self.name,
            overall_score=score,
            severity=RiskSeverity.HIGH if score >= 0.5 else RiskSeverity.MODERATE,
            accident_probability=None,
            recommendation_count=2,
            scenario_coverage=0.0,
            latency_ms=latency,
            notes="Static rules; no accident model or scenario testing",
        )


class IndustryAverageBaseline(CompetitorBaseline):
    """Industry-average static benchmark (consulting report style)."""

    name = "IndustryAverage Report"

    def assess(self, portfolio: AssetPortfolio) -> CompetitorResult:
        import time

        start = time.perf_counter()
        score = 0.35
        latency = (time.perf_counter() - start) * 1000
        return CompetitorResult(
            name=self.name,
            overall_score=score,
            severity=RiskSeverity.MODERATE,
            accident_probability=0.08,
            recommendation_count=3,
            scenario_coverage=0.0,
            latency_ms=latency,
            notes="Generic industry averages; not asset-specific",
        )


class ManualAssessmentBaseline(CompetitorBaseline):
    """Simulates annual manual consultant review."""

    name = "ManualConsultant Review"

    def assess(self, portfolio: AssetPortfolio) -> CompetitorResult:
        import time

        start = time.perf_counter()
        score = 0.28
        if portfolio.building and portfolio.building.last_inspection_years_ago > 3:
            score += 0.1
        latency = (time.perf_counter() - start) * 1000 + 50.0
        return CompetitorResult(
            name=self.name,
            overall_score=min(1.0, score),
            severity=RiskSeverity.MODERATE,
            accident_probability=0.05,
            recommendation_count=4,
            scenario_coverage=0.17,
            latency_ms=latency,
            notes="High latency; limited scenario coverage; annual cadence",
        )


DEFAULT_COMPETITORS: list[CompetitorBaseline] = [
    HeuristicRulesBaseline(),
    IndustryAverageBaseline(),
    ManualAssessmentBaseline(),
]
