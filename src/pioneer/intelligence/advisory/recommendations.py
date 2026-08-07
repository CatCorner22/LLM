"""Business recommendation types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RecommendationPriority(StrEnum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    STRATEGIC = "strategic"
    MONITOR = "monitor"


class RecommendationCategory(StrEnum):
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    INSURANCE = "insurance"
    OPERATIONS = "operations"
    COMPLIANCE = "compliance"
    CAPITAL = "capital"


class BusinessRecommendation(BaseModel):
    id: str
    title: str
    description: str
    category: RecommendationCategory
    priority: RecommendationPriority
    estimated_cost_usd: float | None = None
    risk_reduction: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)


class AdvisoryReport(BaseModel):
    asset_id: str
    executive_summary: str
    overall_risk_score: float
    recommendations: list[BusinessRecommendation]
    scenario_highlights: list[str] = Field(default_factory=list)
    news_signals: list[str] = Field(default_factory=list)
    chemical_signals: list[str] = Field(default_factory=list)
    health_signals: list[str] = Field(default_factory=list)
    governance_signals: list[str] = Field(default_factory=list)
    acquisition_signals: list[str] = Field(default_factory=list)
    drill_recommendations: list[dict[str, object]] = Field(default_factory=list)
    relationship_map: dict[str, object] = Field(default_factory=dict)
