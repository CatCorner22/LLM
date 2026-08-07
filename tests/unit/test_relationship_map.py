"""Unit tests for knowledge relationship map and governance risk."""

from __future__ import annotations

import pytest

from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory
from pioneer.intelligence.relationships.map import (
    KNOWLEDGE_TO_CONTROLS,
    KNOWLEDGE_TO_RISK,
    RelationshipMapBuilder,
)
from pioneer.intelligence.relationships.seeds import GOVERNANCE_KNOWLEDGE_SEEDS
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.governance import GovernanceRiskAssessor
from pioneer.intelligence.risk.models import AssetPortfolio, GovernanceProfile
from pioneer.intelligence.samples import sample_portfolio


@pytest.mark.unit
def test_knowledge_category_crosswalk_includes_governance() -> None:
    assert KnowledgeCategory.EMPLOYEE_CONDUCT in KNOWLEDGE_TO_RISK
    assert KnowledgeCategory.SEGREGATION_OF_DUTIES in KNOWLEDGE_TO_CONTROLS
    assert "employee_conduct" in KNOWLEDGE_TO_CONTROLS[KnowledgeCategory.EMPLOYEE_CONDUCT][0]


@pytest.mark.unit
def test_governance_seeds_cover_conduct_and_sod() -> None:
    categories = {record.category for record in GOVERNANCE_KNOWLEDGE_SEEDS}
    assert KnowledgeCategory.EMPLOYEE_CONDUCT in categories
    assert KnowledgeCategory.SEGREGATION_OF_DUTIES in categories


@pytest.mark.unit
def test_governance_assessor_flags_sod_and_conduct() -> None:
    portfolio = AssetPortfolio(
        governance=GovernanceProfile(
            asset_id="GOV-1",
            conduct_incidents_12m=2,
            policy_training_completion=0.7,
            sod_conflicts=3,
            role_overlap_ratio=0.4,
            dual_control_gaps=2,
        )
    )
    assessment = GovernanceRiskAssessor().assess(portfolio)
    assert assessment is not None
    names = {factor.name for factor in assessment.factors}
    assert "employee_conduct_risk" in names
    assert "segregation_of_duties_risk" in names


@pytest.mark.unit
def test_relationship_map_focus_areas() -> None:
    portfolio = sample_portfolio()
    assessment = CompositeRiskEngine().assess(portfolio)
    rel_map = RelationshipMapBuilder().build(GOVERNANCE_KNOWLEDGE_SEEDS, assessment, portfolio)
    assert rel_map.nodes
    assert rel_map.edges
    focus = rel_map.focus_areas
    assert "employee_conduct" in focus or "segregation_of_duties" in focus
    assert "knowledge_acquisition" in focus
    assert rel_map.acquisition_assessments
    assert rel_map.portfolio_competency is not None


@pytest.mark.unit
def test_composite_includes_governance_when_present() -> None:
    assessment = CompositeRiskEngine().assess(sample_portfolio())
    categories = {factor.category for factor in assessment.factors}
    assert any(cat.value == "governance" for cat in categories)
