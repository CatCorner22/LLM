"""Curated governance knowledge for employee conduct and segregation of duties."""

from __future__ import annotations

from datetime import UTC, datetime

from pioneer.intelligence.ingestion.knowledge import (
    KnowledgeCategory,
    KnowledgeRecord,
    KnowledgeSourceType,
)

GOVERNANCE_KNOWLEDGE_SEEDS: list[KnowledgeRecord] = [
    KnowledgeRecord(
        id="gov:conduct:1",
        source_type=KnowledgeSourceType.EPA_TRI,
        source_id="pioneer_governance",
        title="Workplace harassment and retaliation reporting",
        summary=(
            "Documented conduct incidents correlate with elevated accident and compliance risk. "
            "Maintain anonymous reporting and manager escalation paths."
        ),
        category=KnowledgeCategory.EMPLOYEE_CONDUCT,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        relevance_score=0.85,
        keywords=["conduct", "harassment", "retaliation", "policy"],
        metadata={"framework": "OSHA_general_duty", "control_type": "employee_conduct"},
        url="https://www.osha.gov/workplace-violence",
    ),
    KnowledgeRecord(
        id="gov:conduct:2",
        source_type=KnowledgeSourceType.EPA_TRI,
        source_id="pioneer_governance",
        title="Safety policy training completion thresholds",
        summary=(
            "Sites below 90% annual safety policy attestation show higher near-miss rates. "
            "Conduct refresher training after any reportable incident."
        ),
        category=KnowledgeCategory.EMPLOYEE_CONDUCT,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        relevance_score=0.8,
        keywords=["training", "policy", "safety", "conduct"],
        metadata={"threshold": "0.90", "control_type": "employee_conduct"},
        url="",
    ),
    KnowledgeRecord(
        id="gov:sod:1",
        source_type=KnowledgeSourceType.EPA_TRI,
        source_id="pioneer_governance",
        title="COSO internal control — segregation of duties",
        summary=(
            "No single individual should authorize, record, and custody the same transaction. "
            "Map critical processes and enforce maker-checker controls."
        ),
        category=KnowledgeCategory.SEGREGATION_OF_DUTIES,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        relevance_score=0.9,
        keywords=["segregation", "dual_control", "coso", "internal_control"],
        metadata={"framework": "COSO_2013", "control_type": "segregation_of_duties"},
        url="https://www.coso.org/",
    ),
    KnowledgeRecord(
        id="gov:sod:2",
        source_type=KnowledgeSourceType.EPA_TRI,
        source_id="pioneer_governance",
        title="Privileged access and role overlap review",
        summary=(
            "Quarterly review of shared credentials and overlapping finance/operations roles "
            "reduces fraud and unauthorized change risk."
        ),
        category=KnowledgeCategory.SEGREGATION_OF_DUTIES,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        relevance_score=0.82,
        keywords=["role_overlap", "privileged_access", "dual_control"],
        metadata={"review_cadence": "quarterly", "control_type": "segregation_of_duties"},
        url="",
    ),
]
