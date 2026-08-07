"""Statistical accident and injury likelihood models."""

from __future__ import annotations

import numpy as np

from pioneer.intelligence.risk.competency import competency_exposure_gap
from pioneer.intelligence.risk.models import (
    AssetPortfolio,
    OperationalProfile,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    logistic_probability,
    severity_from_score,
)


class AccidentLikelihoodModel:
    """Logistic models for workplace accident and injury probability."""

    # Feature weights calibrated for interpretability (not clinical claims)
    ACCIDENT_WEIGHTS = np.array([0.9, 0.7, 0.5, 0.6, 0.4, 0.8])
    INJURY_WEIGHTS = np.array([0.6, 0.8, 0.9, 0.5, 0.7, 0.4])
    ACCIDENT_BIAS = -2.2
    INJURY_BIAS = -2.5

    def _feature_vector(
        self,
        portfolio: AssetPortfolio,
        operational: OperationalProfile | None,
    ) -> np.ndarray:
        building_age_norm = 0.3
        structural_risk = 0.3
        weather_risk = 0.2
        pipe_risk = 0.2
        maintenance_norm = 0.3
        incident_norm = 0.0

        if portfolio.building:
            building_age_norm = min(1.0, (2026 - portfolio.building.year_built) / 100.0)
            structural_risk = 1.0 - portfolio.building.structural_condition

        if portfolio.weather:
            weather_risk = (
                portfolio.weather.flood_risk_index * 0.4
                + portfolio.weather.storm_probability_7d * 0.6
            )

        if portfolio.pipes:
            ages = [2026 - pipe.install_year for pipe in portfolio.pipes]
            pipe_risk = min(1.0, max(ages) / 80.0)

        if operational:
            maintenance_norm = min(1.0, operational.maintenance_backlog_days / 30.0)
            incident_norm = min(1.0, operational.prior_incidents_12m / 5.0)

        night_shift = operational.night_shift_ratio if operational else 0.2
        training_gap = 0.0
        if operational:
            training_gap = max(0.0, 1.0 - operational.safety_training_hours / 40.0)

        competency_gap = 0.0
        if portfolio.competency:
            competency_gap = competency_exposure_gap(portfolio.competency)
            if portfolio.competency.history and len(portfolio.competency.history) >= 2:
                ordered = sorted(
                    portfolio.competency.history, key=lambda snapshot: snapshot.captured_at
                )
                transfer_delta = (
                    ordered[-1].scenario_transfer_score - ordered[0].scenario_transfer_score
                )
                if transfer_delta <= -0.05:
                    competency_gap = min(1.0, competency_gap + 0.1)

        exposure = night_shift + max(training_gap, competency_gap) * 0.5

        return np.array(
            [
                building_age_norm,
                structural_risk,
                weather_risk,
                pipe_risk,
                maintenance_norm + incident_norm * 0.5,
                exposure,
            ],
            dtype=float,
        )

    def predict(
        self,
        portfolio: AssetPortfolio,
        operational: OperationalProfile | None = None,
    ) -> RiskAssessment:
        op = operational or portfolio.operational
        features = self._feature_vector(portfolio, op)
        accident_prob = logistic_probability(features, self.ACCIDENT_WEIGHTS, self.ACCIDENT_BIAS)
        injury_prob = logistic_probability(features, self.INJURY_WEIGHTS, self.INJURY_BIAS)

        factors = [
            RiskFactor(
                name="accident_likelihood",
                category=RiskCategory.ACCIDENT,
                score=accident_prob,
                weight=1.0,
                description="Statistical likelihood of a reportable accident (12 months)",
            ),
            RiskFactor(
                name="injury_likelihood",
                category=RiskCategory.ACCIDENT,
                score=injury_prob,
                weight=1.0,
                description="Statistical likelihood of an injury given operational exposure",
            ),
        ]

        overall = max(accident_prob, injury_prob * 0.9)
        asset_id = (
            portfolio.building.asset_id
            if portfolio.building
            else portfolio.pipes[0].asset_id
            if portfolio.pipes
            else "unknown"
        )

        return RiskAssessment(
            asset_id=asset_id,
            overall_score=overall,
            severity=severity_from_score(overall),
            factors=factors,
            accident_probability=accident_prob,
            injury_probability=injury_prob,
            metadata={
                "assessor": "accident_likelihood",
                "model": "logistic_v1",
                "features": ",".join(f"{value:.3f}" for value in features),
            },
        )

    @staticmethod
    def confidence_interval(probability: float, n_samples: int = 100) -> tuple[float, float]:
        """Wilson score interval for displayed uncertainty."""
        if n_samples <= 0:
            return probability, probability
        z = 1.96
        denom = 1 + z**2 / n_samples
        center = probability + z**2 / (2 * n_samples)
        margin = z * np.sqrt((probability * (1 - probability) + z**2 / (4 * n_samples)) / n_samples)
        lower = max(0.0, (center - margin) / denom)
        upper = min(1.0, (center + margin) / denom)
        return float(lower), float(upper)
