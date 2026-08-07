"""Generate diverse, actionable recommendations for business owners."""

from __future__ import annotations

from pioneer.core.logging import get_logger
from pioneer.intelligence.advisory.recommendations import (
    AdvisoryReport,
    BusinessRecommendation,
    RecommendationCategory,
    RecommendationPriority,
)
from pioneer.intelligence.ingestion.feeds import FeedItem
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.models import AssetPortfolio, RiskAssessment, RiskCategory
from pioneer.intelligence.scenarios.runner import AutonomousScenarioRunner, ScenarioRunResult

logger = get_logger(__name__)


class BusinessAdvisor:
    """Translate risk assessments and scenarios into owner-facing guidance."""

    def __init__(
        self,
        engine: CompositeRiskEngine | None = None,
        scenario_runner: AutonomousScenarioRunner | None = None,
    ) -> None:
        self.engine = engine or CompositeRiskEngine()
        self.scenario_runner = scenario_runner or AutonomousScenarioRunner(self.engine)

    def _recommendations_from_assessment(
        self, assessment: RiskAssessment
    ) -> list[BusinessRecommendation]:
        recs: list[BusinessRecommendation] = []
        rec_id = 0

        for factor in assessment.factors:
            if factor.score < 0.35:
                continue
            rec_id += 1

            if factor.category == RiskCategory.BUILDING:
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Schedule structural inspection",
                        description=(
                            "Commission a qualified engineer to inspect load-bearing elements "
                            f"and address '{factor.name}' findings."
                        ),
                        category=RecommendationCategory.SAFETY,
                        priority=RecommendationPriority.IMMEDIATE
                        if factor.score >= 0.6
                        else RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=5000.0,
                        risk_reduction=min(0.3, factor.score * 0.4),
                        rationale=[factor.description, *factor.evidence],
                    )
                )
            elif factor.category == RiskCategory.WEATHER:
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Activate weather contingency plan",
                        description=(
                            "Review flood barriers, drainage, and business continuity plans "
                            f"for {factor.name}."
                        ),
                        category=RecommendationCategory.OPERATIONS,
                        priority=RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=2500.0,
                        risk_reduction=min(0.25, factor.score * 0.35),
                        rationale=[factor.description],
                    )
                )
            elif factor.category == RiskCategory.INFRASTRUCTURE:
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Prioritize pipe replacement segment",
                        description=(
                            "Replace or reline high-risk pipe segments identified by age "
                            f"and Weibull failure analysis ({factor.name})."
                        ),
                        category=RecommendationCategory.MAINTENANCE,
                        priority=RecommendationPriority.IMMEDIATE
                        if factor.score >= 0.55
                        else RecommendationPriority.STRATEGIC,
                        estimated_cost_usd=25000.0,
                        risk_reduction=min(0.4, factor.score * 0.5),
                        rationale=[factor.description, *factor.evidence],
                    )
                )
            elif factor.category == RiskCategory.ACCIDENT:
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Expand safety training and incident review",
                        description=(
                            "Increase safety training hours and conduct a root-cause review "
                            f"to reduce {factor.name}."
                        ),
                        category=RecommendationCategory.SAFETY,
                        priority=RecommendationPriority.IMMEDIATE,
                        estimated_cost_usd=3000.0,
                        risk_reduction=min(0.35, factor.score * 0.45),
                        rationale=[factor.description],
                    )
                )

        if assessment.accident_probability and assessment.accident_probability > 0.15:
            rec_id += 1
            recs.append(
                BusinessRecommendation(
                    id=f"rec_{rec_id}",
                    title="Review insurance and liability coverage",
                    description=(
                        "Statistical accident likelihood exceeds threshold; validate workers' "
                        "comp and liability limits with your broker."
                    ),
                    category=RecommendationCategory.INSURANCE,
                    priority=RecommendationPriority.STRATEGIC,
                    risk_reduction=0.1,
                    rationale=[
                        f"accident_probability={assessment.accident_probability:.3f}",
                        f"injury_probability={assessment.injury_probability or 0:.3f}",
                    ],
                )
            )

        return recs

    def _scenario_highlights(self, scenario_result: ScenarioRunResult) -> list[str]:
        return [
            f"{outcome.scenario_name}: {outcome.business_impact_summary}"
            for outcome in scenario_result.outcomes[:3]
        ]

    def _news_signals(self, feed_items: list[FeedItem]) -> list[str]:
        return [
            f"[{item.source}] {item.title} (relevance={item.relevance_score:.2f})"
            for item in feed_items[:5]
        ]

    def generate_report(
        self,
        portfolio: AssetPortfolio,
        feed_items: list[FeedItem] | None = None,
        *,
        run_scenarios: bool = True,
    ) -> AdvisoryReport:
        assessment = self.engine.assess(portfolio)
        scenario_result = (
            self.scenario_runner.run_autonomous_suite(portfolio) if run_scenarios else None
        )

        recommendations = self._recommendations_from_assessment(assessment)

        if scenario_result and scenario_result.outcomes:
            worst = scenario_result.outcomes[0]
            if worst.delta >= 0.1:
                recommendations.append(
                    BusinessRecommendation(
                        id="rec_scenario",
                        title=f"Mitigate '{worst.scenario_name}' exposure",
                        description=worst.business_impact_summary,
                        category=RecommendationCategory.OPERATIONS,
                        priority=RecommendationPriority.IMMEDIATE
                        if worst.delta >= 0.2
                        else RecommendationPriority.SHORT_TERM,
                        risk_reduction=min(0.3, worst.delta),
                        rationale=[f"scenario_delta={worst.delta:.3f}"],
                    )
                )

        # Deduplicate by title and sort by priority
        priority_order = {
            RecommendationPriority.IMMEDIATE: 0,
            RecommendationPriority.SHORT_TERM: 1,
            RecommendationPriority.STRATEGIC: 2,
            RecommendationPriority.MONITOR: 3,
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 99))

        summary = (
            f"Overall risk score {assessment.overall_score:.2f} ({assessment.severity.value}). "
            f"{len(recommendations)} actionable recommendations identified across safety, "
            "maintenance, operations, and insurance."
        )

        logger.info(
            "advisory_report_generated",
            asset_id=assessment.asset_id,
            recommendations=len(recommendations),
        )

        return AdvisoryReport(
            asset_id=assessment.asset_id,
            executive_summary=summary,
            overall_risk_score=assessment.overall_score,
            recommendations=recommendations,
            scenario_highlights=self._scenario_highlights(scenario_result)
            if scenario_result
            else [],
            news_signals=self._news_signals(feed_items or []),
        )
