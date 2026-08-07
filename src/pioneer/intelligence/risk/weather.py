"""Weather and climate-related operational risk."""

from __future__ import annotations

from pioneer.intelligence.risk.models import (
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    WeatherProfile,
    severity_from_score,
)


class WeatherRiskAssessor:
    """Score weather-driven business disruption and asset damage risk."""

    def assess(self, profile: WeatherProfile) -> RiskAssessment:
        heat_score = min(1.0, profile.heat_wave_days_forecast / 14.0)
        freeze_score = min(1.0, profile.freeze_days_forecast / 14.0)

        factors = [
            RiskFactor(
                name="flood_exposure",
                category=RiskCategory.WEATHER,
                score=profile.flood_risk_index,
                weight=0.3,
                description=f"Regional flood index for {profile.region}",
            ),
            RiskFactor(
                name="wind_exposure",
                category=RiskCategory.WEATHER,
                score=profile.wind_risk_index,
                weight=0.2,
                description="Wind damage potential",
            ),
            RiskFactor(
                name="storm_probability",
                category=RiskCategory.WEATHER,
                score=profile.storm_probability_7d,
                weight=0.25,
                description="7-day severe storm probability",
            ),
            RiskFactor(
                name="extreme_heat",
                category=RiskCategory.WEATHER,
                score=heat_score,
                weight=0.15,
                description="Forecast heat-wave stress days",
            ),
            RiskFactor(
                name="extreme_freeze",
                category=RiskCategory.WEATHER,
                score=freeze_score,
                weight=0.1,
                description="Forecast freeze stress on pipes and structures",
            ),
        ]

        overall = sum(f.score * f.weight for f in factors) / sum(f.weight for f in factors)

        return RiskAssessment(
            asset_id=profile.asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=factors,
            metadata={"assessor": "weather", "region": profile.region},
        )
