"""Structured drill templates for knowledge acquisition remediation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DrillTemplate(BaseModel):
    title: str
    method: str
    estimated_minutes: int = Field(ge=1)
    success_criteria: list[str] = Field(default_factory=list)
    linked_risk_categories: list[str] = Field(default_factory=list)
    confidence_impact: float = Field(ge=0.0, le=1.0, default=0.1)


DRILL_TEMPLATES: dict[str, DrillTemplate] = {
    "scenario_transfer": DrillTemplate(
        title="Novel-condition scenario transfer drill",
        method="scenario_transfer",
        estimated_minutes=45,
        success_criteria=[
            "Correct procedure under changed layout or equipment state",
            "Worker adapts without prompting when one variable changes",
            "Post-drill debrief captures decision rationale",
        ],
        linked_risk_categories=["accident", "operational"],
        confidence_impact=0.18,
    ),
    "explanation_audit": DrillTemplate(
        title="Explain-your-reasoning audit",
        method="explanation_audit",
        estimated_minutes=30,
        success_criteria=[
            "Worker articulates why each step applies, not only what to do",
            "Auditor confirms causal reasoning matches procedure intent",
            "Gaps documented with targeted coaching plan",
        ],
        linked_risk_categories=["governance", "accident"],
        confidence_impact=0.15,
    ),
    "practical_demonstration": DrillTemplate(
        title="Observed practical demonstration",
        method="practical_demonstration",
        estimated_minutes=60,
        success_criteria=[
            "Task completed under realistic conditions with supervisor observation",
            "Performance matches or exceeds certification standard",
            "Certification-only record updated with demonstration evidence",
        ],
        linked_risk_categories=["accident", "operational"],
        confidence_impact=0.2,
    ),
    "skill_decay_refresh": DrillTemplate(
        title="Expertise decay prevention refresh",
        method="scenario_transfer",
        estimated_minutes=40,
        success_criteria=[
            "Transfer score maintained within 10% of prior expert baseline",
            "Novel-condition error rate below site threshold",
            "Refresh logged before skill decay window expires",
        ],
        linked_risk_categories=["operational"],
        confidence_impact=0.12,
    ),
    "peer_validation": DrillTemplate(
        title="Peer validation on high-risk procedure",
        method="practical_demonstration",
        estimated_minutes=50,
        success_criteria=[
            "Independent peer confirms correct execution",
            "Discrepancies resolved before sign-off",
            "Weak-signal skill receives additional scenario coverage",
        ],
        linked_risk_categories=["governance", "accident"],
        confidence_impact=0.14,
    ),
}

SKILL_DRILL_OVERRIDES: dict[str, dict[str, str]] = {
    "lockout_tagout": {
        "scenario_transfer": "LOTO drill with re-ordered isolation points and shared lock box",
        "practical_demonstration": "Live LOTO walkthrough on energized-adjacent equipment",
    },
    "confined_space": {
        "scenario_transfer": "Confined-space entry with changed ventilation and rescue plan",
        "explanation_audit": "Entry supervisor explains atmospheric monitoring rationale",
    },
    "hazmat_handling": {
        "scenario_transfer": "Spill response drill with alternate chemical SDS and PPE set",
        "practical_demonstration": "Observed drum handling and labeling under audit conditions",
    },
}
