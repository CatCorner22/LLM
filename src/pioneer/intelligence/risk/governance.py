"""Governance risk assessor for employee conduct and segregation of duties."""

from __future__ import annotations

from pioneer.intelligence.risk.models import (
    AssetPortfolio,
    GovernanceProfile,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    severity_from_score,
)


class GovernanceRiskAssessor:
    """Assess employee conduct and segregation-of-duties exposure."""

    def assess(self, portfolio: AssetPortfolio) -> RiskAssessment | None:
        gov = portfolio.governance
        if gov is None:
            return None

        conduct_score = self._conduct_score(gov)
        sod_score = self._sod_score(gov)
        factors: list[RiskFactor] = []

        if conduct_score >= 0.2:
            factors.append(
                RiskFactor(
                    name="employee_conduct_risk",
                    category=RiskCategory.GOVERNANCE,
                    score=conduct_score,
                    weight=1.0,
                    description="Employee conduct incidents and policy adherence gaps",
                    evidence=[
                        f"conduct_incidents_12m={gov.conduct_incidents_12m}",
                        f"policy_training_completion={gov.policy_training_completion:.0%}",
                    ],
                )
            )

        if sod_score >= 0.2:
            factors.append(
                RiskFactor(
                    name="segregation_of_duties_risk",
                    category=RiskCategory.GOVERNANCE,
                    score=sod_score,
                    weight=1.0,
                    description="Role overlap and missing dual-control checkpoints",
                    evidence=[
                        f"sod_conflicts={gov.sod_conflicts}",
                        f"role_overlap_ratio={gov.role_overlap_ratio:.2f}",
                        f"dual_control_gaps={gov.dual_control_gaps}",
                    ],
                )
            )

        acquisition_score = self._acquisition_score(portfolio)
        if acquisition_score >= 0.2:
            factors.append(
                RiskFactor(
                    name="knowledge_acquisition_risk",
                    category=RiskCategory.GOVERNANCE,
                    score=acquisition_score,
                    weight=1.0,
                    description=(
                        "Workforce may rely on rote repetition rather than verified expertise"
                    ),
                    evidence=self._acquisition_evidence(portfolio),
                )
            )

        if not factors:
            return None

        overall = max(factor.score for factor in factors)
        asset_id = (
            portfolio.building.asset_id
            if portfolio.building
            else portfolio.operational.asset_id
            if portfolio.operational
            else "unknown"
        )

        return RiskAssessment(
            asset_id=asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=factors,
            metadata={"assessor": "governance"},
        )

    @staticmethod
    def _conduct_score(gov: GovernanceProfile) -> float:
        incident_norm = min(1.0, gov.conduct_incidents_12m / 3.0)
        training_gap = max(0.0, 1.0 - gov.policy_training_completion)
        whistleblower_gap = 0.15 if not gov.whistleblower_channel else 0.0
        return min(1.0, incident_norm * 0.5 + training_gap * 0.35 + whistleblower_gap)

    @staticmethod
    def _sod_score(gov: GovernanceProfile) -> float:
        conflict_norm = min(1.0, gov.sod_conflicts / 5.0)
        overlap = min(1.0, gov.role_overlap_ratio)
        dual_gap = min(1.0, gov.dual_control_gaps / 4.0)
        return min(1.0, conflict_norm * 0.4 + overlap * 0.35 + dual_gap * 0.25)

    @staticmethod
    def _acquisition_score(portfolio: AssetPortfolio) -> float:
        from pioneer.intelligence.relationships.acquisition import (
            KnowledgeAcquisitionAssessor,
            acquisition_risk_score,
        )

        competency = portfolio.competency
        if competency is None:
            return 0.0
        assessor = KnowledgeAcquisitionAssessor()
        if competency.skills:
            report = assessor.assess_portfolio(competency)
            aggregate_score = acquisition_risk_score(report.aggregate)
            if report.rote_skills:
                skill_scores = [
                    acquisition_risk_score(item)
                    for item in report.skill_assessments
                    if item.skill_id in report.rote_skills
                ]
                weakest = max(skill_scores) if skill_scores else aggregate_score
                return min(1.0, aggregate_score * 0.6 + weakest * 0.4)
            return aggregate_score
        return acquisition_risk_score(assessor.assess(competency))

    @staticmethod
    def _acquisition_evidence(portfolio: AssetPortfolio) -> list[str]:
        from pioneer.intelligence.relationships.acquisition import KnowledgeAcquisitionAssessor

        competency = portfolio.competency
        if competency is None:
            return []
        assessor = KnowledgeAcquisitionAssessor()
        if competency.skills:
            report = assessor.assess_portfolio(competency)
            evidence = [
                f"aggregate_level={report.aggregate.expertise_level.value}",
                f"expertise_score={report.aggregate.expertise_score:.2f}",
                f"rote_score={report.aggregate.rote_repetition_score:.2f}",
                f"team_coverage={report.team_coverage:.0%}",
            ]
            if report.rote_skills:
                evidence.append(f"rote_skills={','.join(report.rote_skills)}")
            if report.weakest_skills:
                evidence.append(f"weakest_skills={','.join(report.weakest_skills)}")
            if report.aggregate.trend:
                evidence.append(f"trend={report.aggregate.trend.value}")
            return evidence
        result = assessor.assess(competency)
        return [
            f"expertise_level={result.expertise_level.value}",
            f"expertise_score={result.expertise_score:.2f}",
            f"rote_score={result.rote_repetition_score:.2f}",
        ]
