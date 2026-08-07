"""Lightweight competency exposure helpers for risk models."""

from __future__ import annotations

from pioneer.intelligence.competency_models import CompetencyProfile


def competency_exposure_gap(profile: CompetencyProfile) -> float:
    """Estimate workforce understanding gap from competency signals."""
    resilience = max(0.0, 1.0 - profile.novel_condition_error_rate)
    expertise = min(
        1.0,
        0.35 * profile.scenario_transfer_score
        + 0.25 * profile.explanation_audit_score
        + 0.25 * profile.practical_demonstration_rate
        + 0.15 * resilience,
    )
    gap = max(0.0, 1.0 - expertise)
    completion_gap = max(0.0, profile.training_completion_rate - expertise)
    cert_gap = max(0.0, profile.certification_only_ratio - profile.practical_demonstration_rate)
    rote = min(1.0, completion_gap * 0.6 + cert_gap * 0.4)
    if rote >= 0.25:
        gap = min(1.0, gap + rote * 0.35)
    return gap
