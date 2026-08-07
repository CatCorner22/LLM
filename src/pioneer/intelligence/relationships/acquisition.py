"""Assess whether workforce knowledge reflects expertise or rote repetition."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from pioneer.intelligence.competency_models import (
    CompetencyProfile,
    CompetencySnapshot,
    SkillCompetency,
)
from pioneer.intelligence.relationships.drills import DRILL_TEMPLATES, SKILL_DRILL_OVERRIDES


class ExpertiseLevel(StrEnum):
    EXPERT = "expert"
    DEVELOPING = "developing"
    ROTE_REPETITION = "rote_repetition"
    UNVERIFIED = "unverified"


class TrendDirection(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECAYING = "decaying"


class AcquisitionSignal(BaseModel):
    """Observable indicator used to distinguish understanding from memorization."""

    name: str
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    interpretation: str = ""


class ConfidenceBreakdown(BaseModel):
    """Explainable confidence reasoning for competency verdicts."""

    overall: float = Field(ge=0.0, le=1.0)
    assessment_volume: float = Field(ge=0.0, le=1.0)
    signal_coverage: float = Field(ge=0.0, le=1.0)
    signal_agreement: float = Field(ge=0.0, le=1.0)
    trend_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    limiting_factors: list[str] = Field(default_factory=list)
    expertise_interval: tuple[float, float] | None = None


class DrillRecommendation(BaseModel):
    """Structured remediation drill tied to a weak acquisition signal."""

    drill_id: str
    skill_id: str | None = None
    title: str
    method: str
    priority: str
    estimated_minutes: int
    success_criteria: list[str] = Field(default_factory=list)
    linked_risk_categories: list[str] = Field(default_factory=list)
    confidence_impact: float = Field(ge=0.0, le=1.0, default=0.1)
    rationale: str = ""


class CompetencyAssessment(BaseModel):
    """Verdict on whether demonstrated knowledge reflects true expertise."""

    subject_id: str
    expertise_level: ExpertiseLevel
    expertise_score: float = Field(ge=0.0, le=1.0)
    rote_repetition_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[AcquisitionSignal] = Field(default_factory=list)
    summary: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    skill_id: str | None = None
    trend: TrendDirection | None = None
    confidence_breakdown: ConfidenceBreakdown | None = None
    drills: list[DrillRecommendation] = Field(default_factory=list)
    weakest_signal: str | None = None


class PortfolioCompetencyReport(BaseModel):
    """Multi-skill competency assessment with team-level summary."""

    asset_id: str
    aggregate: CompetencyAssessment
    skill_assessments: list[CompetencyAssessment] = Field(default_factory=list)
    weakest_skills: list[str] = Field(default_factory=list)
    expert_skills: list[str] = Field(default_factory=list)
    rote_skills: list[str] = Field(default_factory=list)
    team_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    all_drills: list[DrillRecommendation] = Field(default_factory=list)


class KnowledgeAcquisitionAssessor:
    """Evaluate competency profiles using transfer, explanation, and error signals."""

    TRANSFER_WEIGHT = 0.35
    EXPLANATION_WEIGHT = 0.25
    PRACTICAL_WEIGHT = 0.25
    NOVEL_RESILIENCE_WEIGHT = 0.15

    ROTE_GAP_THRESHOLD = 0.25
    EXPERT_SCORE_THRESHOLD = 0.75
    ROTE_SCORE_THRESHOLD = 0.55
    DECAY_PENALTY = 0.12
    TREND_DELTA_THRESHOLD = 0.05

    SIGNAL_NAMES = (
        "scenario_transfer",
        "explanation_audit",
        "practical_demonstration",
        "novel_condition_resilience",
    )

    def assess_portfolio(self, profile: CompetencyProfile) -> PortfolioCompetencyReport:
        """Assess aggregate and per-skill competency with drill recommendations."""
        skill_assessments = [
            self.assess(
                self._skill_to_profile(profile.asset_id, skill),
                skill_id=skill.skill_id,
            )
            for skill in profile.skills
        ]
        aggregate = self.assess(profile, history=profile.history)
        aggregate.subject_id = profile.asset_id

        weakest = sorted(
            skill_assessments,
            key=lambda item: (item.expertise_score, -item.rote_repetition_score),
        )
        expert_skills = [
            item.skill_id
            for item in skill_assessments
            if item.expertise_level == ExpertiseLevel.EXPERT
        ]
        rote_skills = [
            item.skill_id
            for item in skill_assessments
            if item.expertise_level == ExpertiseLevel.ROTE_REPETITION
        ]
        all_drills = self._dedupe_drills(
            aggregate.drills + [drill for item in skill_assessments for drill in item.drills]
        )
        verified = [
            item for item in skill_assessments if item.expertise_level != ExpertiseLevel.UNVERIFIED
        ]
        coverage = len(verified) / len(skill_assessments) if skill_assessments else 1.0

        return PortfolioCompetencyReport(
            asset_id=profile.asset_id,
            aggregate=aggregate,
            skill_assessments=skill_assessments,
            weakest_skills=[item.skill_id for item in weakest[:3] if item.skill_id],
            expert_skills=[skill for skill in expert_skills if skill],
            rote_skills=[skill for skill in rote_skills if skill],
            team_coverage=round(coverage, 4),
            all_drills=all_drills,
        )

    def assess(
        self,
        profile: CompetencyProfile,
        *,
        skill_id: str | None = None,
        history: list[CompetencySnapshot] | None = None,
    ) -> CompetencyAssessment:
        trend = self._compute_trend(history or profile.history)
        signals = self._build_signals(profile, trend)
        expertise_score = self._expertise_score(profile)
        if trend == TrendDirection.DECAYING:
            expertise_score = max(0.0, expertise_score - self.DECAY_PENALTY)
        rote_score = self._rote_repetition_score(profile, expertise_score)
        level = self._classify(profile, expertise_score, rote_score, trend)
        confidence_breakdown = self._confidence_breakdown(profile, signals, trend)
        weakest = self._weakest_signal(profile)
        drills = self._generate_drills(level, weakest, trend, skill_id)

        subject_id = skill_id or profile.asset_id
        return CompetencyAssessment(
            subject_id=subject_id,
            skill_id=skill_id,
            expertise_level=level,
            expertise_score=round(expertise_score, 4),
            rote_repetition_score=round(rote_score, 4),
            confidence=round(confidence_breakdown.overall, 4),
            signals=signals,
            summary=self._summary(level, expertise_score, rote_score, trend, skill_id),
            recommended_actions=self._recommended_actions(level, trend, drills),
            trend=trend,
            confidence_breakdown=confidence_breakdown,
            drills=drills,
            weakest_signal=weakest,
        )

    def _skill_to_profile(self, asset_id: str, skill: SkillCompetency) -> CompetencyProfile:
        return CompetencyProfile(
            asset_id=asset_id,
            training_completion_rate=skill.training_completion_rate,
            scenario_transfer_score=skill.scenario_transfer_score,
            explanation_audit_score=skill.explanation_audit_score,
            novel_condition_error_rate=skill.novel_condition_error_rate,
            practical_demonstration_rate=skill.practical_demonstration_rate,
            certification_only_ratio=skill.certification_only_ratio,
            assessment_count=skill.assessment_count,
        )

    def _build_signals(
        self,
        profile: CompetencyProfile,
        trend: TrendDirection | None,
    ) -> list[AcquisitionSignal]:
        signals = [
            AcquisitionSignal(
                name="scenario_transfer",
                value=profile.scenario_transfer_score,
                weight=self.TRANSFER_WEIGHT,
                interpretation="Applies knowledge to novel conditions",
            ),
            AcquisitionSignal(
                name="explanation_audit",
                value=profile.explanation_audit_score,
                weight=self.EXPLANATION_WEIGHT,
                interpretation="Can explain why, not only what",
            ),
            AcquisitionSignal(
                name="practical_demonstration",
                value=profile.practical_demonstration_rate,
                weight=self.PRACTICAL_WEIGHT,
                interpretation="Observed performance vs certification only",
            ),
            AcquisitionSignal(
                name="novel_condition_resilience",
                value=max(0.0, 1.0 - profile.novel_condition_error_rate),
                weight=self.NOVEL_RESILIENCE_WEIGHT,
                interpretation="Maintains accuracy when context changes",
            ),
            AcquisitionSignal(
                name="training_completion_gap",
                value=max(0.0, profile.training_completion_rate - profile.scenario_transfer_score),
                weight=0.2,
                interpretation="Large gap suggests repetition without understanding",
            ),
        ]
        if trend == TrendDirection.DECAYING:
            signals.append(
                AcquisitionSignal(
                    name="skill_decay",
                    value=0.75,
                    weight=0.15,
                    interpretation="Transfer and explanation scores declining over time",
                )
            )
        elif trend == TrendDirection.IMPROVING:
            signals.append(
                AcquisitionSignal(
                    name="skill_growth",
                    value=0.8,
                    weight=0.1,
                    interpretation="Competency improving across historical snapshots",
                )
            )
        return signals

    def _expertise_score(self, profile: CompetencyProfile) -> float:
        resilience = max(0.0, 1.0 - profile.novel_condition_error_rate)
        return min(
            1.0,
            self.TRANSFER_WEIGHT * profile.scenario_transfer_score
            + self.EXPLANATION_WEIGHT * profile.explanation_audit_score
            + self.PRACTICAL_WEIGHT * profile.practical_demonstration_rate
            + self.NOVEL_RESILIENCE_WEIGHT * resilience,
        )

    def _rote_repetition_score(self, profile: CompetencyProfile, expertise_score: float) -> float:
        completion_gap = max(0.0, profile.training_completion_rate - expertise_score)
        cert_without_practice = max(
            0.0, profile.certification_only_ratio - profile.practical_demonstration_rate
        )
        return min(1.0, completion_gap * 0.6 + cert_without_practice * 0.4)

    def _classify(
        self,
        profile: CompetencyProfile,
        expertise_score: float,
        rote_score: float,
        trend: TrendDirection | None,
    ) -> ExpertiseLevel:
        if profile.assessment_count < 1:
            return ExpertiseLevel.UNVERIFIED
        rote_pattern = (
            profile.training_completion_rate >= 0.85
            and expertise_score < self.ROTE_SCORE_THRESHOLD
            and rote_score >= self.ROTE_GAP_THRESHOLD
        ) or (expertise_score < 0.5 and rote_score >= self.ROTE_GAP_THRESHOLD)
        if rote_pattern:
            return ExpertiseLevel.ROTE_REPETITION
        if expertise_score >= self.EXPERT_SCORE_THRESHOLD and rote_score < self.ROTE_GAP_THRESHOLD:
            return (
                ExpertiseLevel.DEVELOPING
                if trend == TrendDirection.DECAYING
                else ExpertiseLevel.EXPERT
            )
        if expertise_score >= 0.5:
            return ExpertiseLevel.DEVELOPING
        return ExpertiseLevel.UNVERIFIED

    def _confidence_breakdown(
        self,
        profile: CompetencyProfile,
        signals: list[AcquisitionSignal],
        trend: TrendDirection | None,
    ) -> ConfidenceBreakdown:
        assessment_volume = min(1.0, profile.assessment_count / 5.0)
        signal_coverage = sum(1 for signal in signals if signal.value > 0.0) / max(len(signals), 1)
        weighted_values = [signal.value for signal in signals if signal.name in self.SIGNAL_NAMES]
        if weighted_values:
            mean_value = sum(weighted_values) / len(weighted_values)
            variance = sum((value - mean_value) ** 2 for value in weighted_values) / len(
                weighted_values
            )
            signal_agreement = max(0.0, 1.0 - variance * 4.0)
        else:
            signal_agreement = 0.0

        limiting_factors: list[str] = []
        if profile.assessment_count < 2:
            limiting_factors.append("Fewer than 2 independent assessments recorded")
        if profile.certification_only_ratio > 0.6 and profile.practical_demonstration_rate < 0.5:
            limiting_factors.append("Heavy certification reliance without observed performance")
        if not profile.history and trend is None:
            limiting_factors.append("No historical snapshots for trend validation")

        trend_confidence = None
        if profile.history and len(profile.history) >= 2:
            trend_confidence = min(1.0, len(profile.history) / 4.0)

        overall = min(
            1.0,
            assessment_volume * 0.35
            + signal_coverage * 0.25
            + signal_agreement * 0.25
            + (trend_confidence or 0.5) * 0.15,
        )
        expertise_interval = self._wilson_interval(
            self._expertise_score(profile),
            n_samples=max(1, profile.assessment_count * 10),
        )

        return ConfidenceBreakdown(
            overall=overall,
            assessment_volume=assessment_volume,
            signal_coverage=signal_coverage,
            signal_agreement=signal_agreement,
            trend_confidence=trend_confidence,
            limiting_factors=limiting_factors,
            expertise_interval=expertise_interval,
        )

    @staticmethod
    def _compute_trend(history: list[CompetencySnapshot]) -> TrendDirection | None:
        if len(history) < 2:
            return None
        ordered = sorted(history, key=lambda snapshot: snapshot.captured_at)
        first, last = ordered[0], ordered[-1]
        transfer_delta = last.scenario_transfer_score - first.scenario_transfer_score
        explanation_delta = last.explanation_audit_score - first.explanation_audit_score
        avg_delta = (transfer_delta + explanation_delta) / 2.0
        if avg_delta >= KnowledgeAcquisitionAssessor.TREND_DELTA_THRESHOLD:
            return TrendDirection.IMPROVING
        if avg_delta <= -KnowledgeAcquisitionAssessor.TREND_DELTA_THRESHOLD:
            return TrendDirection.DECAYING
        return TrendDirection.STABLE

    @staticmethod
    def _weakest_signal(profile: CompetencyProfile) -> str:
        scored = {
            "scenario_transfer": profile.scenario_transfer_score,
            "explanation_audit": profile.explanation_audit_score,
            "practical_demonstration": profile.practical_demonstration_rate,
            "novel_condition_resilience": max(0.0, 1.0 - profile.novel_condition_error_rate),
        }
        return min(scored, key=scored.get)  # type: ignore[arg-type]

    def _generate_drills(
        self,
        level: ExpertiseLevel,
        weakest_signal: str,
        trend: TrendDirection | None,
        skill_id: str | None,
    ) -> list[DrillRecommendation]:
        drills: list[DrillRecommendation] = []
        if level == ExpertiseLevel.ROTE_REPETITION:
            drills.extend(
                self._drill_from_template(
                    "scenario_transfer", "immediate", skill_id, weakest_signal
                )
            )
            drills.extend(
                self._drill_from_template(
                    "explanation_audit", "immediate", skill_id, weakest_signal
                )
            )
            drills.extend(
                self._drill_from_template(
                    "practical_demonstration", "short_term", skill_id, weakest_signal
                )
            )
        elif level == ExpertiseLevel.DEVELOPING:
            method = weakest_signal if weakest_signal in DRILL_TEMPLATES else "scenario_transfer"
            drills.extend(self._drill_from_template(method, "short_term", skill_id, weakest_signal))
            drills.extend(
                self._drill_from_template("peer_validation", "short_term", skill_id, weakest_signal)
            )
        elif level == ExpertiseLevel.EXPERT and trend == TrendDirection.DECAYING:
            drills.extend(
                self._drill_from_template(
                    "skill_decay_refresh", "short_term", skill_id, weakest_signal
                )
            )
        elif level == ExpertiseLevel.EXPERT:
            drills.extend(
                self._drill_from_template(
                    "skill_decay_refresh", "monitor", skill_id, weakest_signal
                )
            )
        elif level == ExpertiseLevel.UNVERIFIED:
            drills.extend(
                self._drill_from_template(
                    "scenario_transfer", "short_term", skill_id, weakest_signal
                )
            )
            drills.extend(
                self._drill_from_template(
                    "explanation_audit", "short_term", skill_id, weakest_signal
                )
            )
        return drills

    def _drill_from_template(
        self,
        template_key: str,
        priority: str,
        skill_id: str | None,
        weakest_signal: str,
    ) -> list[DrillRecommendation]:
        template = DRILL_TEMPLATES.get(template_key)
        if not template:
            return []
        title = template.title
        if skill_id and skill_id in SKILL_DRILL_OVERRIDES:
            override = SKILL_DRILL_OVERRIDES[skill_id].get(template_key)
            if override:
                title = override
        drill_id = f"drill:{template_key}:{skill_id or 'aggregate'}"
        return [
            DrillRecommendation(
                drill_id=drill_id,
                skill_id=skill_id,
                title=title,
                method=template.method,
                priority=priority,
                estimated_minutes=template.estimated_minutes,
                success_criteria=list(template.success_criteria),
                linked_risk_categories=list(template.linked_risk_categories),
                confidence_impact=template.confidence_impact,
                rationale=f"Targets weakest signal: {weakest_signal}",
            )
        ]

    @staticmethod
    def _dedupe_drills(drills: list[DrillRecommendation]) -> list[DrillRecommendation]:
        seen: set[str] = set()
        unique: list[DrillRecommendation] = []
        for drill in drills:
            if drill.drill_id in seen:
                continue
            seen.add(drill.drill_id)
            unique.append(drill)
        return unique

    @staticmethod
    def _summary(
        level: ExpertiseLevel,
        expertise_score: float,
        rote_score: float,
        trend: TrendDirection | None,
        skill_id: str | None,
    ) -> str:
        prefix = f"Skill '{skill_id}': " if skill_id else ""
        trend_note = ""
        if trend == TrendDirection.DECAYING:
            trend_note = " Skill decay detected — refresh assessments urgently."
        elif trend == TrendDirection.IMPROVING:
            trend_note = " Competency trending upward."

        if level == ExpertiseLevel.EXPERT:
            return (
                f"{prefix}Demonstrates transferable expertise (score={expertise_score:.2f}). "
                "Knowledge holds under novel conditions and explanation audits." + trend_note
            )
        if level == ExpertiseLevel.ROTE_REPETITION:
            return (
                f"{prefix}High completion with weak transfer (rote score={rote_score:.2f}). "
                "Likely repetition without deep understanding — add scenario checks." + trend_note
            )
        if level == ExpertiseLevel.DEVELOPING:
            return (
                f"{prefix}Mixed acquisition signals (expertise={expertise_score:.2f}). "
                "Continue structured assessments before granting expert status." + trend_note
            )
        return f"{prefix}Insufficient competency assessment data to verify expertise." + trend_note

    @staticmethod
    def _recommended_actions(
        level: ExpertiseLevel,
        trend: TrendDirection | None,
        drills: list[DrillRecommendation],
    ) -> list[str]:
        actions = [drill.title for drill in drills[:3]]
        if level == ExpertiseLevel.ROTE_REPETITION and not actions:
            actions = [
                "Run scenario-transfer drills with changed conditions",
                "Require explain-your-reasoning audits on critical tasks",
                "Pair certifications with observed practical demonstrations",
            ]
        elif level == ExpertiseLevel.DEVELOPING and not actions:
            actions = [
                "Increase novel-condition scenario coverage",
                "Use peer validation on high-risk procedures",
            ]
        elif level == ExpertiseLevel.EXPERT and trend == TrendDirection.DECAYING:
            actions = ["Schedule decay-prevention refresh before expert status lapses"]
        elif level == ExpertiseLevel.EXPERT and not actions:
            actions = ["Maintain periodic transfer tests to prevent skill decay"]
        elif level == ExpertiseLevel.UNVERIFIED and not actions:
            actions = [
                "Establish baseline transfer and explanation assessments",
                "Track novel-condition error rates alongside training completion",
            ]
        return actions

    @staticmethod
    def _wilson_interval(probability: float, n_samples: int = 100) -> tuple[float, float]:
        if n_samples <= 0:
            return probability, probability
        z = 1.96
        denom = 1 + z**2 / n_samples
        center = probability + z**2 / (2 * n_samples)
        margin = z * ((probability * (1 - probability) + z**2 / (4 * n_samples)) / n_samples) ** 0.5
        lower = max(0.0, (center - margin) / denom)
        upper = min(1.0, (center + margin) / denom)
        return float(lower), float(upper)


def acquisition_risk_score(assessment: CompetencyAssessment) -> float:
    """Map competency assessment to a governance risk score in [0, 1]."""
    if assessment.expertise_level == ExpertiseLevel.ROTE_REPETITION:
        return min(1.0, 0.45 + assessment.rote_repetition_score * 0.55)
    if assessment.expertise_level == ExpertiseLevel.UNVERIFIED:
        return 0.42
    if assessment.expertise_level == ExpertiseLevel.DEVELOPING:
        return max(0.2, 0.55 - assessment.expertise_score * 0.35)
    if assessment.trend == TrendDirection.DECAYING:
        return max(0.25, 0.5 - assessment.expertise_score * 0.25)
    return max(0.0, 0.3 - assessment.expertise_score * 0.3)
