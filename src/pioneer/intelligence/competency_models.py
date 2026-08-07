"""Competency profile types shared across risk and relationship modules."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SkillCompetency(BaseModel):
    """Per-skill competency signals for multi-skill workforce assessment."""

    skill_id: str
    role: str | None = None
    criticality: float = Field(default=0.5, ge=0.0, le=1.0)
    training_completion_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    scenario_transfer_score: float = Field(default=0.7, ge=0.0, le=1.0)
    explanation_audit_score: float = Field(default=0.7, ge=0.0, le=1.0)
    novel_condition_error_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    practical_demonstration_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    certification_only_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    assessment_count: int = Field(default=1, ge=0)


class CompetencySnapshot(BaseModel):
    """Point-in-time competency metrics for trend analysis."""

    captured_at: datetime
    training_completion_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    scenario_transfer_score: float = Field(default=0.7, ge=0.0, le=1.0)
    explanation_audit_score: float = Field(default=0.7, ge=0.0, le=1.0)
    novel_condition_error_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    practical_demonstration_rate: float = Field(default=0.75, ge=0.0, le=1.0)


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
    skills: list[SkillCompetency] = Field(default_factory=list)
    history: list[CompetencySnapshot] = Field(default_factory=list)
