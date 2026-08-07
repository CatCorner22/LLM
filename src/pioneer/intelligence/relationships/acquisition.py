"""Assess whether workforce knowledge reflects expertise or rote repetition."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from pioneer.intelligence.risk.models import CompetencyProfile


class ExpertiseLevel(StrEnum):
    EXPERT = "expert"
    DEVELOPING = "developing"
    ROTE_REPETITION = "rote_repetition"
    UNVERIFIED = "unverified"


class AcquisitionSignal(BaseModel):
    """Observable indicator used to distinguish understanding from memorization."""

    name: str
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    interpretation: str = ""


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


class KnowledgeAcquisitionAssessor:
    """Evaluate competency profiles using transfer, explanation, and error signals."""

    TRANSFER_WEIGHT = 0.35
    EXPLANATION_WEIGHT = 0.25
    PRACTICAL_WEIGHT = 0.25
    NOVEL_RESILIENCE_WEIGHT = 0.15

    ROTE_GAP_THRESHOLD = 0.25
    EXPERT_SCORE_THRESHOLD = 0.75
    ROTE_SCORE_THRESHOLD = 0.55

    def assess(self, profile: CompetencyProfile) -> CompetencyAssessment:
        signals = self._build_signals(profile)
        expertise_score = self._expertise_score(profile)
        rote_score = self._rote_repetition_score(profile, expertise_score)
        level = self._classify(profile, expertise_score, rote_score)
        confidence = self._confidence(profile, signals)

        return CompetencyAssessment(
            subject_id=profile.asset_id,
            expertise_level=level,
            expertise_score=round(expertise_score, 4),
            rote_repetition_score=round(rote_score, 4),
            confidence=round(confidence, 4),
            signals=signals,
            summary=self._summary(level, expertise_score, rote_score),
            recommended_actions=self._recommended_actions(level),
        )

    def _build_signals(self, profile: CompetencyProfile) -> list[AcquisitionSignal]:
        return [
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
    ) -> ExpertiseLevel:
        if profile.assessment_count < 1:
            return ExpertiseLevel.UNVERIFIED
        if (
            profile.training_completion_rate >= 0.85
            and expertise_score < self.ROTE_SCORE_THRESHOLD
            and rote_score >= self.ROTE_GAP_THRESHOLD
        ):
            return ExpertiseLevel.ROTE_REPETITION
        if expertise_score >= self.EXPERT_SCORE_THRESHOLD and rote_score < self.ROTE_GAP_THRESHOLD:
            return ExpertiseLevel.EXPERT
        if expertise_score >= 0.5:
            return ExpertiseLevel.DEVELOPING
        if rote_score >= self.ROTE_GAP_THRESHOLD:
            return ExpertiseLevel.ROTE_REPETITION
        return ExpertiseLevel.UNVERIFIED

    @staticmethod
    def _confidence(profile: CompetencyProfile, signals: list[AcquisitionSignal]) -> float:
        base = min(1.0, profile.assessment_count / 3.0)
        signal_coverage = sum(1 for signal in signals if signal.value > 0.0) / len(signals)
        return min(1.0, base * 0.6 + signal_coverage * 0.4)

    @staticmethod
    def _summary(level: ExpertiseLevel, expertise_score: float, rote_score: float) -> str:
        if level == ExpertiseLevel.EXPERT:
            return (
                f"Demonstrates transferable expertise (score={expertise_score:.2f}). "
                "Knowledge holds under novel conditions and explanation audits."
            )
        if level == ExpertiseLevel.ROTE_REPETITION:
            return (
                f"High completion with weak transfer (rote score={rote_score:.2f}). "
                "Likely repetition without deep understanding — add scenario checks."
            )
        if level == ExpertiseLevel.DEVELOPING:
            return (
                f"Mixed acquisition signals (expertise={expertise_score:.2f}). "
                "Continue structured assessments before granting expert status."
            )
        return "Insufficient competency assessment data to verify expertise."

    @staticmethod
    def _recommended_actions(level: ExpertiseLevel) -> list[str]:
        if level == ExpertiseLevel.ROTE_REPETITION:
            return [
                "Run scenario-transfer drills with changed conditions",
                "Require explain-your-reasoning audits on critical tasks",
                "Pair certifications with observed practical demonstrations",
            ]
        if level == ExpertiseLevel.DEVELOPING:
            return [
                "Increase novel-condition scenario coverage",
                "Use peer validation on high-risk procedures",
            ]
        if level == ExpertiseLevel.EXPERT:
            return ["Maintain periodic transfer tests to prevent skill decay"]
        return [
            "Establish baseline transfer and explanation assessments",
            "Track novel-condition error rates alongside training completion",
        ]
