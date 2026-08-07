"""Infrastructure and pipe-age failure risk using Weibull reliability."""

from __future__ import annotations

from datetime import UTC, datetime

from pioneer.intelligence.risk.models import (
    PipeProfile,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    severity_from_score,
    weibull_failure_probability,
)

# Material-specific Weibull parameters (shape, scale years)
MATERIAL_WEIBULL: dict[str, tuple[float, float]] = {
    "cast_iron": (2.8, 50.0),
    "ductile_iron": (3.0, 70.0),
    "pvc": (2.2, 80.0),
    "steel": (3.5, 60.0),
    "concrete": (2.5, 65.0),
    "lead": (2.0, 40.0),
}


class InfrastructureRiskAssessor:
    """Assess pipe and utility infrastructure failure risk from age and condition."""

    def assess_pipe(self, profile: PipeProfile) -> RiskAssessment:
        current_year = datetime.now(UTC).year
        age = max(0, current_year - profile.install_year)
        shape, scale = MATERIAL_WEIBULL.get(profile.material.lower(), (2.5, 55.0))
        failure_prob = weibull_failure_probability(age, shape, scale)

        pressure_stress = min(1.0, profile.pressure_bar / 10.0)
        corrosion_score = profile.soil_corrosivity
        inspection_gap = 1.0 - profile.inspection_score

        factors = [
            RiskFactor(
                name="pipe_age_failure",
                category=RiskCategory.INFRASTRUCTURE,
                score=failure_prob,
                weight=0.35,
                description=f"Weibull failure probability at {age} years ({profile.material})",
                evidence=[f"install_year={profile.install_year}", f"age={age}"],
            ),
            RiskFactor(
                name="pressure_stress",
                category=RiskCategory.INFRASTRUCTURE,
                score=pressure_stress,
                weight=0.15,
                description="Operating pressure relative to design limits",
            ),
            RiskFactor(
                name="soil_corrosivity",
                category=RiskCategory.INFRASTRUCTURE,
                score=corrosion_score,
                weight=0.2,
                description="Soil-driven corrosion exposure",
            ),
            RiskFactor(
                name="inspection_condition",
                category=RiskCategory.INFRASTRUCTURE,
                score=inspection_gap,
                weight=0.3,
                description="Gap between actual and target inspection condition",
            ),
        ]

        overall = sum(f.score * f.weight for f in factors) / sum(f.weight for f in factors)

        return RiskAssessment(
            asset_id=profile.asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=factors,
            metadata={
                "assessor": "infrastructure",
                "pipe_age_years": str(age),
                "material": profile.material,
                "weibull_failure_prob": f"{failure_prob:.4f}",
            },
        )

    def assess_portfolio_pipes(self, pipes: list[PipeProfile]) -> RiskAssessment | None:
        if not pipes:
            return None
        assessments = [self.assess_pipe(pipe) for pipe in pipes]
        max_assessment = max(assessments, key=lambda item: item.overall_score)
        combined_factors: list[RiskFactor] = []
        for assessment in assessments:
            combined_factors.extend(assessment.factors)
        avg_score = sum(a.overall_score for a in assessments) / len(assessments)
        blended = max(avg_score, max_assessment.overall_score * 0.85)
        return RiskAssessment(
            asset_id=pipes[0].asset_id,
            overall_score=blended,
            severity=severity_from_score(blended),
            factors=combined_factors,
            metadata={
                "assessor": "infrastructure_portfolio",
                "pipe_count": str(len(pipes)),
                "worst_pipe": max_assessment.asset_id,
            },
        )
