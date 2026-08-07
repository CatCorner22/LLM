"""Unit tests for knowledge acquisition assessment."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory
from pioneer.intelligence.relationships.acquisition import (
    ExpertiseLevel,
    KnowledgeAcquisitionAssessor,
    TrendDirection,
    acquisition_risk_score,
)
from pioneer.intelligence.relationships.map import RelationshipMapBuilder
from pioneer.intelligence.relationships.seeds import GOVERNANCE_KNOWLEDGE_SEEDS
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.governance import GovernanceRiskAssessor
from pioneer.intelligence.risk.models import (
    CompetencyProfile,
    CompetencySnapshot,
)
from pioneer.intelligence.samples import expert_competency_profile, sample_portfolio
from pioneer.intelligence.scenarios.definitions import DEFAULT_SCENARIOS
from pioneer.intelligence.scenarios.runner import AutonomousScenarioRunner


@pytest.mark.unit
def test_assessor_classifies_rote_repetition() -> None:
    profile = CompetencyProfile(
        asset_id="ROTE-1",
        training_completion_rate=0.95,
        scenario_transfer_score=0.42,
        explanation_audit_score=0.38,
        novel_condition_error_rate=0.48,
        practical_demonstration_rate=0.35,
        certification_only_ratio=0.72,
        assessment_count=3,
    )
    result = KnowledgeAcquisitionAssessor().assess(profile)
    assert result.expertise_level == ExpertiseLevel.ROTE_REPETITION
    assert result.rote_repetition_score > result.expertise_score
    assert result.drills
    assert result.confidence_breakdown is not None
    assert result.weakest_signal is not None


@pytest.mark.unit
def test_assessor_classifies_expert() -> None:
    profile = expert_competency_profile()
    result = KnowledgeAcquisitionAssessor().assess(profile)
    assert result.expertise_level == ExpertiseLevel.EXPERT
    assert result.expertise_score >= 0.75
    assert result.rote_repetition_score < 0.25


@pytest.mark.unit
def test_assessor_detects_skill_decay() -> None:
    profile = CompetencyProfile(
        asset_id="DECAY-1",
        training_completion_rate=0.95,
        scenario_transfer_score=0.8,
        explanation_audit_score=0.82,
        novel_condition_error_rate=0.1,
        practical_demonstration_rate=0.88,
        certification_only_ratio=0.15,
        assessment_count=4,
        history=[
            CompetencySnapshot(
                captured_at=datetime(2025, 1, 1, tzinfo=UTC),
                scenario_transfer_score=0.92,
                explanation_audit_score=0.9,
            ),
            CompetencySnapshot(
                captured_at=datetime(2026, 1, 1, tzinfo=UTC),
                scenario_transfer_score=0.8,
                explanation_audit_score=0.82,
            ),
        ],
    )
    result = KnowledgeAcquisitionAssessor().assess(profile)
    assert result.trend == TrendDirection.DECAYING
    assert result.expertise_level == ExpertiseLevel.DEVELOPING
    assert any(signal.name == "skill_decay" for signal in result.signals)


@pytest.mark.unit
def test_assessor_portfolio_multi_skill() -> None:
    portfolio = sample_portfolio()
    assert portfolio.competency is not None
    report = KnowledgeAcquisitionAssessor().assess_portfolio(portfolio.competency)
    assert len(report.skill_assessments) == 3
    assert "lockout_tagout" in report.rote_skills
    assert "confined_space" in report.expert_skills
    assert report.all_drills
    assert report.aggregate.trend == TrendDirection.DECAYING


@pytest.mark.unit
def test_acquisition_risk_score_varies_by_level() -> None:
    assessor = KnowledgeAcquisitionAssessor()
    rote = assessor.assess(
        CompetencyProfile(
            asset_id="R",
            training_completion_rate=0.95,
            scenario_transfer_score=0.4,
            explanation_audit_score=0.35,
            novel_condition_error_rate=0.5,
            practical_demonstration_rate=0.3,
            certification_only_ratio=0.75,
            assessment_count=2,
        )
    )
    expert = assessor.assess(expert_competency_profile("E"))
    assert acquisition_risk_score(rote) > acquisition_risk_score(expert)


@pytest.mark.unit
def test_governance_includes_acquisition_factor() -> None:
    assessment = GovernanceRiskAssessor().assess(sample_portfolio())
    assert assessment is not None
    names = {factor.name for factor in assessment.factors}
    assert "knowledge_acquisition_risk" in names


@pytest.mark.unit
def test_composite_risk_reflects_competency_gaps() -> None:
    weak = CompositeRiskEngine().assess(sample_portfolio())
    strong = CompositeRiskEngine().assess(
        sample_portfolio().model_copy(update={"competency": expert_competency_profile("BLDG-001")})
    )
    assert weak.overall_score >= strong.overall_score


@pytest.mark.unit
def test_competency_stress_scenario_increases_risk() -> None:
    portfolio = sample_portfolio()
    scenario = next(item for item in DEFAULT_SCENARIOS if item.id == "scn_competency_stress")
    result = AutonomousScenarioRunner().run(portfolio, scenarios=[scenario])
    assert result.outcomes
    assert result.outcomes[0].delta >= 0.0


@pytest.mark.unit
def test_relationship_map_includes_acquisition_assessments() -> None:
    portfolio = sample_portfolio()
    assessment = CompositeRiskEngine().assess(portfolio)
    rel_map = RelationshipMapBuilder().build(GOVERNANCE_KNOWLEDGE_SEEDS, assessment, portfolio)
    assert len(rel_map.acquisition_assessments) >= 4
    assert rel_map.portfolio_competency is not None
    node_types = {node.node_type for node in rel_map.nodes}
    assert "acquisition_assessment" in node_types


@pytest.mark.unit
def test_relationship_map_knowledge_acquisition_focus() -> None:
    portfolio = sample_portfolio()
    assessment = CompositeRiskEngine().assess(portfolio)
    rel_map = RelationshipMapBuilder().build(GOVERNANCE_KNOWLEDGE_SEEDS, assessment, portfolio)
    assert "knowledge_acquisition" in rel_map.focus_areas


@pytest.mark.unit
def test_acquisition_seeds_present() -> None:
    categories = {record.category for record in GOVERNANCE_KNOWLEDGE_SEEDS}
    assert KnowledgeCategory.KNOWLEDGE_ACQUISITION in categories
