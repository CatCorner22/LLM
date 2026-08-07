"""Building structural and occupancy risk assessment."""

from __future__ import annotations

from datetime import UTC, datetime

from pioneer.intelligence.risk.models import (
    BuildingProfile,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    severity_from_score,
)


class BuildingRiskAssessor:
    """Score building risk from age, condition, occupancy, and seismic exposure."""

    def assess(self, profile: BuildingProfile) -> RiskAssessment:
        current_year = datetime.now(UTC).year
        age = max(0, current_year - profile.year_built)
        age_score = min(1.0, age / 100.0)
        condition_score = 1.0 - profile.structural_condition
        occupancy_density = profile.occupancy / max(profile.floor_area_sqm / 10.0, 1.0)
        occupancy_score = min(1.0, occupancy_density / 5.0)
        seismic_score = profile.seismic_zone / 4.0
        inspection_score = min(1.0, profile.last_inspection_years_ago / 5.0)
        fire_score = 0.0 if profile.fire_suppression else 0.35

        factors = [
            RiskFactor(
                name="building_age",
                category=RiskCategory.BUILDING,
                score=age_score,
                weight=0.2,
                description=f"Building age {age} years",
                evidence=[f"year_built={profile.year_built}"],
            ),
            RiskFactor(
                name="structural_condition",
                category=RiskCategory.BUILDING,
                score=condition_score,
                weight=0.25,
                description="Inverse of structural condition index",
            ),
            RiskFactor(
                name="occupancy_density",
                category=RiskCategory.BUILDING,
                score=occupancy_score,
                weight=0.15,
                description="Occupants per effective floor unit",
            ),
            RiskFactor(
                name="seismic_exposure",
                category=RiskCategory.BUILDING,
                score=seismic_score,
                weight=0.15,
                description=f"Seismic zone {profile.seismic_zone}",
            ),
            RiskFactor(
                name="inspection_recency",
                category=RiskCategory.BUILDING,
                score=inspection_score,
                weight=0.15,
                description="Time since last structural inspection",
            ),
            RiskFactor(
                name="fire_suppression",
                category=RiskCategory.BUILDING,
                score=fire_score,
                weight=0.1,
                description="Missing fire suppression increases severity",
            ),
        ]

        weighted = sum(f.score * f.weight for f in factors)
        weight_total = sum(f.weight for f in factors)
        overall = weighted / weight_total if weight_total else 0.0

        return RiskAssessment(
            asset_id=profile.asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=factors,
            metadata={"assessor": "building", "age_years": str(age)},
        )
