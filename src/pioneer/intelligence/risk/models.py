"""Shared risk intelligence types and statistical utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field


class RiskSeverity(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(StrEnum):
    BUILDING = "building"
    WEATHER = "weather"
    INFRASTRUCTURE = "infrastructure"
    ACCIDENT = "accident"
    OPERATIONAL = "operational"
    GOVERNANCE = "governance"


class RiskFactor(BaseModel):
    """Single scored risk contributor."""

    name: str
    category: RiskCategory
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    description: str = ""
    evidence: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Composite risk assessment for an asset or portfolio."""

    asset_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    severity: RiskSeverity
    factors: list[RiskFactor]
    accident_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    injury_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)


class BuildingProfile(BaseModel):
    asset_id: str
    year_built: int = Field(ge=1800, le=2100)
    occupancy: int = Field(default=50, ge=0)
    floor_area_sqm: float = Field(default=1000.0, gt=0)
    structural_condition: float = Field(default=0.8, ge=0.0, le=1.0)
    fire_suppression: bool = False
    seismic_zone: int = Field(default=1, ge=0, le=4)
    last_inspection_years_ago: float = Field(default=2.0, ge=0.0)


class PipeProfile(BaseModel):
    asset_id: str
    install_year: int = Field(ge=1850, le=2100)
    material: str = "cast_iron"
    diameter_mm: float = Field(default=150.0, gt=0)
    pressure_bar: float = Field(default=4.0, gt=0)
    soil_corrosivity: float = Field(default=0.3, ge=0.0, le=1.0)
    inspection_score: float = Field(default=0.7, ge=0.0, le=1.0)


class WeatherProfile(BaseModel):
    asset_id: str
    region: str
    flood_risk_index: float = Field(default=0.2, ge=0.0, le=1.0)
    wind_risk_index: float = Field(default=0.2, ge=0.0, le=1.0)
    heat_wave_days_forecast: int = Field(default=0, ge=0)
    freeze_days_forecast: int = Field(default=0, ge=0)
    storm_probability_7d: float = Field(default=0.1, ge=0.0, le=1.0)


class OperationalProfile(BaseModel):
    asset_id: str
    worker_count: int = Field(default=10, ge=0)
    safety_training_hours: float = Field(default=8.0, ge=0.0)
    prior_incidents_12m: int = Field(default=0, ge=0)
    maintenance_backlog_days: float = Field(default=5.0, ge=0.0)
    night_shift_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class GovernanceProfile(BaseModel):
    """Employee conduct and segregation-of-duties indicators."""

    asset_id: str
    conduct_incidents_12m: int = Field(default=0, ge=0)
    policy_training_completion: float = Field(default=0.95, ge=0.0, le=1.0)
    whistleblower_channel: bool = True
    sod_conflicts: int = Field(default=0, ge=0)
    role_overlap_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    dual_control_gaps: int = Field(default=0, ge=0)


class CompetencyProfile(BaseModel):
    """Signals to distinguish expertise from rote repetition."""

    asset_id: str
    training_completion_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    scenario_transfer_score: float = Field(default=0.7, ge=0.0, le=1.0)
    explanation_audit_score: float = Field(default=0.7, ge=0.0, le=1.0)
    novel_condition_error_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    practical_demonstration_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    certification_only_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    assessment_count: int = Field(default=1, ge=0)


class AssetPortfolio(BaseModel):
    """Full asset context for composite risk scoring."""

    building: BuildingProfile | None = None
    pipes: list[PipeProfile] = Field(default_factory=list)
    weather: WeatherProfile | None = None
    operational: OperationalProfile | None = None
    governance: GovernanceProfile | None = None
    competency: CompetencyProfile | None = None


def logistic_probability(features: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Compute logistic regression probability."""
    logit = float(np.dot(features, weights) + bias)
    return float(1.0 / (1.0 + np.exp(-logit)))


def weibull_failure_probability(age_years: float, shape: float, scale: float) -> float:
    """Weibull CDF — probability of failure by given age."""
    if age_years <= 0:
        return 0.0
    return float(1.0 - np.exp(-((age_years / scale) ** shape)))


def severity_from_score(score: float) -> RiskSeverity:
    if score >= 0.75:
        return RiskSeverity.CRITICAL
    if score >= 0.5:
        return RiskSeverity.HIGH
    if score >= 0.25:
        return RiskSeverity.MODERATE
    return RiskSeverity.LOW
