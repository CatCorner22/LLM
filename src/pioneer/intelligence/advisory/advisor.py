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
from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory, KnowledgeRecord
from pioneer.intelligence.ingestion.scheduler import KnowledgeStore
from pioneer.intelligence.relationships.acquisition import (
    CompetencyAssessment,
    DrillRecommendation,
    ExpertiseLevel,
    TrendDirection,
)
from pioneer.intelligence.relationships.map import RelationshipMapBuilder
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
            elif factor.category == RiskCategory.GOVERNANCE:
                if "conduct" in factor.name:
                    recs.append(
                        BusinessRecommendation(
                            id=f"rec_{rec_id}",
                            title="Strengthen employee conduct program",
                            description=(
                                "Increase policy attestation, refresh code-of-conduct training, "
                                "and review open conduct incidents with HR and legal."
                            ),
                            category=RecommendationCategory.COMPLIANCE,
                            priority=RecommendationPriority.IMMEDIATE
                            if factor.score >= 0.55
                            else RecommendationPriority.SHORT_TERM,
                            estimated_cost_usd=1500.0,
                            risk_reduction=min(0.3, factor.score * 0.4),
                            rationale=[factor.description, *factor.evidence],
                            evidence_sources=["employee_conduct"],
                        )
                    )
                elif "segregation" in factor.name:
                    recs.append(
                        BusinessRecommendation(
                            id=f"rec_{rec_id}",
                            title="Remediate segregation-of-duties conflicts",
                            description=(
                                "Split conflicting roles, enforce maker-checker on approvals, "
                                "and close dual-control gaps on financial and safety workflows."
                            ),
                            category=RecommendationCategory.COMPLIANCE,
                            priority=RecommendationPriority.IMMEDIATE
                            if factor.score >= 0.5
                            else RecommendationPriority.STRATEGIC,
                            estimated_cost_usd=4000.0,
                            risk_reduction=min(0.35, factor.score * 0.45),
                            rationale=[factor.description, *factor.evidence],
                            evidence_sources=["segregation_of_duties"],
                        )
                    )
                elif "acquisition" in factor.name or "knowledge" in factor.name:
                    recs.append(
                        BusinessRecommendation(
                            id=f"rec_{rec_id}",
                            title="Close workforce understanding gaps",
                            description=(
                                "Competency assessments indicate rote repetition or unverified "
                                "expertise. Deploy scenario-transfer drills and explanation audits "
                                "before relying on training completion metrics alone."
                            ),
                            category=RecommendationCategory.OPERATIONS,
                            priority=RecommendationPriority.IMMEDIATE
                            if factor.score >= 0.5
                            else RecommendationPriority.SHORT_TERM,
                            estimated_cost_usd=2500.0,
                            risk_reduction=min(0.35, factor.score * 0.45),
                            rationale=[factor.description, *factor.evidence],
                            evidence_sources=["knowledge_acquisition"],
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

    def _recommendations_from_acquisition(
        self,
        assessments: list[CompetencyAssessment],
        drills: list[DrillRecommendation],
        start_id: int,
    ) -> list[BusinessRecommendation]:
        recs: list[BusinessRecommendation] = []
        rec_id = start_id
        for assessment in assessments:
            if assessment.expertise_level == ExpertiseLevel.ROTE_REPETITION:
                rec_id += 1
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Validate workforce understanding beyond training completion",
                        description=(
                            "High training completion with weak transfer and explanation "
                            "scores indicates rote repetition without understanding. "
                            + " ".join(assessment.recommended_actions)
                        ),
                        category=RecommendationCategory.OPERATIONS,
                        priority=RecommendationPriority.IMMEDIATE
                        if assessment.rote_repetition_score >= 0.45
                        else RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=2000.0,
                        risk_reduction=min(0.3, assessment.rote_repetition_score * 0.5),
                        rationale=[assessment.summary, *assessment.recommended_actions],
                        evidence_sources=[
                            "knowledge_acquisition",
                            assessment.skill_id or "aggregate",
                        ],
                    )
                )
            elif assessment.expertise_level == ExpertiseLevel.UNVERIFIED:
                rec_id += 1
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Establish competency baselines before expert designation",
                        description=(
                            "Insufficient transfer, explanation, and practical demonstration "
                            "data to verify expertise. " + " ".join(assessment.recommended_actions)
                        ),
                        category=RecommendationCategory.COMPLIANCE,
                        priority=RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=1500.0,
                        risk_reduction=0.15,
                        rationale=[assessment.summary],
                        evidence_sources=["knowledge_acquisition"],
                    )
                )
            elif assessment.expertise_level == ExpertiseLevel.DEVELOPING:
                rec_id += 1
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title=(
                            f"Strengthen {assessment.weakest_signal or 'competency'} "
                            "before expert sign-off"
                        ),
                        description=(
                            f"Mixed acquisition signals for {assessment.skill_id or 'workforce'}. "
                            + " ".join(assessment.recommended_actions)
                        ),
                        category=RecommendationCategory.OPERATIONS,
                        priority=RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=1800.0,
                        risk_reduction=0.2,
                        rationale=[assessment.summary],
                        evidence_sources=["knowledge_acquisition"],
                    )
                )
            elif (
                assessment.expertise_level == ExpertiseLevel.EXPERT
                and assessment.trend == TrendDirection.DECAYING
            ):
                rec_id += 1
                recs.append(
                    BusinessRecommendation(
                        id=f"rec_{rec_id}",
                        title="Prevent expert skill decay",
                        description=(
                            "Historical snapshots show declining transfer and explanation scores. "
                            "Schedule refresh drills before expert designation lapses."
                        ),
                        category=RecommendationCategory.OPERATIONS,
                        priority=RecommendationPriority.SHORT_TERM,
                        estimated_cost_usd=1200.0,
                        risk_reduction=0.15,
                        rationale=[assessment.summary],
                        evidence_sources=["knowledge_acquisition"],
                    )
                )

        for drill in drills[:5]:
            rec_id += 1
            priority = RecommendationPriority.IMMEDIATE
            if drill.priority == "short_term":
                priority = RecommendationPriority.SHORT_TERM
            elif drill.priority == "monitor":
                priority = RecommendationPriority.MONITOR
            recs.append(
                BusinessRecommendation(
                    id=f"rec_{rec_id}",
                    title=drill.title,
                    description=(
                        f"{drill.rationale} Success criteria: "
                        + "; ".join(drill.success_criteria[:2])
                    ),
                    category=RecommendationCategory.OPERATIONS,
                    priority=priority,
                    estimated_cost_usd=float(drill.estimated_minutes * 2),
                    risk_reduction=min(0.25, drill.confidence_impact),
                    rationale=drill.success_criteria,
                    evidence_sources=["knowledge_acquisition", drill.drill_id],
                )
            )
        return recs

    def _acquisition_signals(self, assessments: list[CompetencyAssessment]) -> list[str]:
        signals: list[str] = []
        for item in assessments:
            skill_label = f"skill={item.skill_id} " if item.skill_id else ""
            trend_label = f" trend={item.trend.value}" if item.trend else ""
            confidence_note = ""
            if item.confidence_breakdown and item.confidence_breakdown.limiting_factors:
                confidence_note = (
                    f" limits={'; '.join(item.confidence_breakdown.limiting_factors[:2])}"
                )
            signals.append(
                f"[{item.subject_id}] {skill_label}{item.expertise_level.value}: {item.summary} "
                f"(expertise={item.expertise_score:.2f}, rote={item.rote_repetition_score:.2f}, "
                f"confidence={item.confidence:.2f}{trend_label}{confidence_note})"
            )
        return signals

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

    def _knowledge_signals(
        self, records: list[KnowledgeRecord], category: KnowledgeCategory, limit: int = 5
    ) -> list[str]:
        filtered = [record for record in records if record.category == category]
        filtered.sort(key=lambda record: record.relevance_score, reverse=True)
        return [
            f"[{record.source_id}] {record.title} (relevance={record.relevance_score:.2f})"
            for record in filtered[:limit]
        ]

    def generate_report(
        self,
        portfolio: AssetPortfolio,
        feed_items: list[FeedItem] | None = None,
        knowledge_records: list[KnowledgeRecord] | None = None,
        *,
        run_scenarios: bool = True,
        load_knowledge: bool = True,
    ) -> AdvisoryReport:
        assessment = self.engine.assess(portfolio)
        scenario_result = (
            self.scenario_runner.run_autonomous_suite(portfolio) if run_scenarios else None
        )

        records = knowledge_records
        if records is None and load_knowledge:
            store = KnowledgeStore()
            store.seed_governance_knowledge()
            records = store.load_all_knowledge(limit=50)
        records = records or []

        relationship_map = RelationshipMapBuilder().build(records, assessment, portfolio)

        recommendations = self._recommendations_from_assessment(assessment)
        portfolio_report = relationship_map.portfolio_competency
        all_drills = portfolio_report.all_drills if portfolio_report else []
        recommendations.extend(
            self._recommendations_from_acquisition(
                relationship_map.acquisition_assessments,
                all_drills,
                len(recommendations),
            )
        )

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
            "maintenance, operations, governance, and insurance."
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
            chemical_signals=self._knowledge_signals(records, KnowledgeCategory.CHEMICAL),
            health_signals=self._knowledge_signals(records, KnowledgeCategory.HEALTH),
            governance_signals=self._governance_signals(records),
            acquisition_signals=self._acquisition_signals(relationship_map.acquisition_assessments),
            drill_recommendations=[drill.model_dump() for drill in all_drills[:10]],
            relationship_map=relationship_map.model_dump(),
        )

    def _governance_signals(self, records: list[KnowledgeRecord]) -> list[str]:
        conduct = self._knowledge_signals(records, KnowledgeCategory.EMPLOYEE_CONDUCT, limit=3)
        sod = self._knowledge_signals(records, KnowledgeCategory.SEGREGATION_OF_DUTIES, limit=3)
        return conduct + sod
