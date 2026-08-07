"""Unit tests for knowledge acquisition assessment."""

from __future__ import annotations

import pytest

from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory
from pioneer.intelligence.relationships.acquisition import (
    ExpertiseLevel,
    KnowledgeAcquisitionAssessor,
)
from pioneer.intelligence.relationships.map import RelationshipMapBuilder
from pioneer.intelligence.relationships.seeds import GOVERNANCE_KNOWLEDGE_SEEDS
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.models import CompetencyProfile
from pioneer.intelligence.samples import sample_portfolio


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
    assert "rote" in result.summary.lower() or "transfer" in result.summary.lower()
    assert len(result.recommended_actions) >= 2


@pytest.mark.unit
def test_assessor_classifies_expert() -> None:
    profile = CompetencyProfile(
        asset_id="EXP-1",
        training_completion_rate=0.95,
        scenario_transfer_score=0.88,
        explanation_audit_score=0.9,
        novel_condition_error_rate=0.08,
        practical_demonstration_rate=0.92,
        certification_only_ratio=0.15,
        assessment_count=4,
    )
    result = KnowledgeAcquisitionAssessor().assess(profile)
    assert result.expertise_level == ExpertiseLevel.EXPERT
    assert result.expertise_score >= 0.75
    assert result.rote_repetition_score < 0.25


@pytest.mark.unit
def test_assessor_unverified_without_assessments() -> None:
    profile = CompetencyProfile(
        asset_id="UNV-1",
        assessment_count=0,
    )
    result = KnowledgeAcquisitionAssessor().assess(profile)
    assert result.expertise_level == ExpertiseLevel.UNVERIFIED


@pytest.mark.unit
def test_relationship_map_includes_acquisition_assessments() -> None:
    portfolio = sample_portfolio()
    assessment = CompositeRiskEngine().assess(portfolio)
    rel_map = RelationshipMapBuilder().build(GOVERNANCE_KNOWLEDGE_SEEDS, assessment, portfolio)
    assert rel_map.acquisition_assessments
    competency = rel_map.acquisition_assessments[0]
    assert competency.expertise_level == ExpertiseLevel.ROTE_REPETITION
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
