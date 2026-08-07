"""Relationship map linking knowledge categories to risks and controls."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from pioneer.intelligence.advisory.recommendations import RecommendationCategory
from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory, KnowledgeRecord
from pioneer.intelligence.risk.models import RiskAssessment, RiskCategory


class RelationshipType(StrEnum):
    INFORMS = "informs"
    MITIGATES = "mitigates"
    REQUIRES = "requires"
    ELEVATES = "elevates"
    CONTROLS = "controls"


class ControlDomain(StrEnum):
    EMPLOYEE_CONDUCT = "employee_conduct"
    SEGREGATION_OF_DUTIES = "segregation_of_duties"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    FINANCIAL = "financial"


class MapNode(BaseModel):
    id: str
    label: str
    node_type: str
    category: str


class MapEdge(BaseModel):
    source: str
    target: str
    relationship: RelationshipType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class KnowledgeRelationshipMap(BaseModel):
    """Graph of knowledge categories, risk domains, and control requirements."""

    nodes: list[MapNode] = Field(default_factory=list)
    edges: list[MapEdge] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)


KNOWLEDGE_TO_RISK: dict[KnowledgeCategory, list[RiskCategory]] = {
    KnowledgeCategory.NEWS: [
        RiskCategory.BUILDING,
        RiskCategory.WEATHER,
        RiskCategory.INFRASTRUCTURE,
    ],
    KnowledgeCategory.CHEMICAL: [RiskCategory.ACCIDENT, RiskCategory.OPERATIONAL],
    KnowledgeCategory.HEALTH: [RiskCategory.ACCIDENT, RiskCategory.OPERATIONAL],
    KnowledgeCategory.REGULATORY: [RiskCategory.GOVERNANCE, RiskCategory.OPERATIONAL],
    KnowledgeCategory.INDUSTRY: [RiskCategory.OPERATIONAL, RiskCategory.INFRASTRUCTURE],
    KnowledgeCategory.EMPLOYEE_CONDUCT: [RiskCategory.GOVERNANCE, RiskCategory.ACCIDENT],
    KnowledgeCategory.SEGREGATION_OF_DUTIES: [
        RiskCategory.GOVERNANCE,
        RiskCategory.OPERATIONAL,
    ],
}

KNOWLEDGE_TO_CONTROLS: dict[KnowledgeCategory, list[ControlDomain]] = {
    KnowledgeCategory.NEWS: [ControlDomain.OPERATIONS, ControlDomain.SAFETY],
    KnowledgeCategory.CHEMICAL: [ControlDomain.SAFETY, ControlDomain.COMPLIANCE],
    KnowledgeCategory.HEALTH: [ControlDomain.SAFETY, ControlDomain.OPERATIONS],
    KnowledgeCategory.REGULATORY: [ControlDomain.COMPLIANCE, ControlDomain.FINANCIAL],
    KnowledgeCategory.INDUSTRY: [ControlDomain.OPERATIONS],
    KnowledgeCategory.EMPLOYEE_CONDUCT: [
        ControlDomain.EMPLOYEE_CONDUCT,
        ControlDomain.COMPLIANCE,
    ],
    KnowledgeCategory.SEGREGATION_OF_DUTIES: [
        ControlDomain.SEGREGATION_OF_DUTIES,
        ControlDomain.FINANCIAL,
    ],
}

CONTROL_TO_RECOMMENDATIONS: dict[ControlDomain, list[RecommendationCategory]] = {
    ControlDomain.EMPLOYEE_CONDUCT: [
        RecommendationCategory.COMPLIANCE,
        RecommendationCategory.SAFETY,
    ],
    ControlDomain.SEGREGATION_OF_DUTIES: [
        RecommendationCategory.COMPLIANCE,
        RecommendationCategory.OPERATIONS,
    ],
    ControlDomain.SAFETY: [RecommendationCategory.SAFETY],
    ControlDomain.COMPLIANCE: [RecommendationCategory.COMPLIANCE],
    ControlDomain.OPERATIONS: [RecommendationCategory.OPERATIONS],
    ControlDomain.FINANCIAL: [
        RecommendationCategory.CAPITAL,
        RecommendationCategory.COMPLIANCE,
    ],
}


def _catalog_nodes() -> dict[str, MapNode]:
    nodes: dict[str, MapNode] = {}
    for category in KnowledgeCategory:
        node_id = f"knowledge:{category.value}"
        nodes[node_id] = MapNode(
            id=node_id,
            label=category.value.replace("_", " ").title(),
            node_type="knowledge",
            category=category.value,
        )
    for risk in RiskCategory:
        node_id = f"risk:{risk.value}"
        nodes[node_id] = MapNode(
            id=node_id,
            label=risk.value.replace("_", " ").title(),
            node_type="risk",
            category=risk.value,
        )
    for control in ControlDomain:
        node_id = f"control:{control.value}"
        nodes[node_id] = MapNode(
            id=node_id,
            label=control.value.replace("_", " ").title(),
            node_type="control",
            category=control.value,
        )
    return nodes


def _catalog_edges() -> list[MapEdge]:
    edges: list[MapEdge] = []
    for knowledge_cat, risk_cats in KNOWLEDGE_TO_RISK.items():
        source = f"knowledge:{knowledge_cat.value}"
        for risk_cat in risk_cats:
            edges.append(
                MapEdge(
                    source=source,
                    target=f"risk:{risk_cat.value}",
                    relationship=RelationshipType.INFORMS,
                    weight=0.7,
                    rationale=f"{knowledge_cat.value} informs {risk_cat.value}",
                )
            )
    for knowledge_cat, controls in KNOWLEDGE_TO_CONTROLS.items():
        source = f"knowledge:{knowledge_cat.value}"
        for control in controls:
            edges.append(
                MapEdge(
                    source=source,
                    target=f"control:{control.value}",
                    relationship=RelationshipType.REQUIRES,
                    weight=0.8,
                    rationale=f"{knowledge_cat.value} requires {control.value}",
                )
            )
    return edges


def _record_edges(
    records: list[KnowledgeRecord],
    nodes: dict[str, MapNode],
) -> list[MapEdge]:
    edges: list[MapEdge] = []
    for record in records:
        record_id = f"knowledge_record:{record.id}"
        for risk_cat in KNOWLEDGE_TO_RISK.get(record.category, []):
            edges.append(
                MapEdge(
                    source=record_id,
                    target=f"risk:{risk_cat.value}",
                    relationship=RelationshipType.ELEVATES,
                    weight=record.relevance_score,
                    rationale=record.title,
                )
            )
        if record_id not in nodes:
            nodes[record_id] = MapNode(
                id=record_id,
                label=record.title[:80],
                node_type="knowledge_record",
                category=record.category.value,
            )
    return edges


def _assessment_edges(
    assessment: RiskAssessment,
    nodes: dict[str, MapNode],
) -> list[MapEdge]:
    edges: list[MapEdge] = []
    for factor in assessment.factors:
        if factor.score < 0.35:
            continue
        factor_id = f"risk_factor:{factor.name}"
        nodes[factor_id] = MapNode(
            id=factor_id,
            label=factor.name,
            node_type="risk_factor",
            category=factor.category.value,
        )
        edges.append(
            MapEdge(
                source=factor_id,
                target=f"risk:{factor.category.value}",
                relationship=RelationshipType.ELEVATES,
                weight=factor.score,
                rationale=factor.description,
            )
        )
        if factor.category != RiskCategory.GOVERNANCE:
            continue
        edges.extend(
            [
                MapEdge(
                    source=factor_id,
                    target="control:employee_conduct",
                    relationship=RelationshipType.REQUIRES,
                    weight=factor.score,
                    rationale="Governance weakness requires conduct controls",
                ),
                MapEdge(
                    source=factor_id,
                    target="control:segregation_of_duties",
                    relationship=RelationshipType.REQUIRES,
                    weight=factor.score,
                    rationale="Governance weakness requires SoD controls",
                ),
            ]
        )
    return edges


class RelationshipMapBuilder:
    """Build a relationship map from knowledge records and a risk assessment."""

    FOCUS_AREAS = ("employee_conduct", "segregation_of_duties")

    def build(
        self,
        records: list[KnowledgeRecord],
        assessment: RiskAssessment | None = None,
    ) -> KnowledgeRelationshipMap:
        nodes = _catalog_nodes()
        edges = _catalog_edges()
        edges.extend(_record_edges(records, nodes))
        if assessment:
            edges.extend(_assessment_edges(assessment, nodes))

        present_categories = {record.category for record in records}
        focus = [
            area
            for area in self.FOCUS_AREAS
            if _focus_triggered(area, present_categories, assessment)
        ]
        return KnowledgeRelationshipMap(
            nodes=list(nodes.values()),
            edges=edges,
            focus_areas=focus,
        )


def _has_governance_factor(assessment: RiskAssessment, keyword: str) -> bool:
    return any(
        factor.category == RiskCategory.GOVERNANCE and keyword in factor.name
        for factor in assessment.factors
    )


def _focus_triggered(
    area: str,
    categories: set[KnowledgeCategory],
    assessment: RiskAssessment | None,
) -> bool:
    if area == "employee_conduct":
        if KnowledgeCategory.EMPLOYEE_CONDUCT in categories:
            return True
        return assessment is not None and _has_governance_factor(assessment, "conduct")
    if area == "segregation_of_duties":
        if KnowledgeCategory.SEGREGATION_OF_DUTIES in categories:
            return True
        return assessment is not None and _has_governance_factor(assessment, "segregation")
    return False
