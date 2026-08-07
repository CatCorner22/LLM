"""Unified knowledge base records for feeds, chemical inventory, and health data."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeSourceType(StrEnum):
    RSS = "rss"
    EPA_TRI = "epa_tri"
    CDC_NCHS = "cdc_nchs"


class KnowledgeCategory(StrEnum):
    NEWS = "news"
    CHEMICAL = "chemical"
    HEALTH = "health"
    REGULATORY = "regulatory"
    INDUSTRY = "industry"
    EMPLOYEE_CONDUCT = "employee_conduct"
    SEGREGATION_OF_DUTIES = "segregation_of_duties"


class KnowledgeRecord(BaseModel):
    """Canonical record stored in the Pioneer Intelligence knowledge base."""

    id: str
    source_type: KnowledgeSourceType
    source_id: str
    title: str
    summary: str
    category: KnowledgeCategory
    published_at: datetime
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    url: str = ""


def chemical_relevance(row: dict[str, Any]) -> tuple[float, list[str]]:
    """Score EPA TRI chemical inventory rows for risk relevance."""
    keywords: list[str] = []
    score = 0.35

    if row.get("carc_ind") == "1":
        score += 0.25
        keywords.append("carcinogen")
    if row.get("caac_ind") == "1":
        score += 0.1
        keywords.append("air_toxic")
    if row.get("pfas_ind") == "1":
        score += 0.2
        keywords.append("pfas")
    if row.get("pbt_ind") == "1":
        score += 0.15
        keywords.append("persistent_toxic")
    if row.get("metal_ind") == "1":
        score += 0.05
        keywords.append("metal")

    name = str(row.get("chem_name", "")).lower()
    for term in ("formaldehyde", "benzene", "asbestos", "lead", "mercury", "chlorine"):
        if term in name:
            score += 0.05
            keywords.append(term)

    return min(1.0, score), sorted(set(keywords))


def er_visit_relevance(row: dict[str, Any]) -> tuple[float, list[str]]:
    """Score CDC emergency department visit statistics for risk relevance."""
    keywords: list[str] = []
    measure = str(row.get("measure", "")).lower()
    subgroup = str(row.get("subgroup", "")).lower()
    score = 0.3

    if "injury" in measure or "poison" in measure:
        score += 0.35
        keywords.extend(["injury", "poisoning"])
    if "circulatory" in measure or "respiratory" in measure:
        score += 0.15
        keywords.append("health")
    if subgroup == "all visits" and row.get("leading_10_ranking") == "0":
        score += 0.1
        keywords.append("total_volume")

    try:
        estimate = float(row.get("estimate", 0))
        if estimate > 10_000_000:
            score += 0.05
    except (TypeError, ValueError):
        pass

    return min(1.0, score), sorted(set(keywords))


def chemical_record_from_row(row: dict[str, Any], index: int) -> KnowledgeRecord:
    """Convert an EPA TRI_CHEM_INFO row into a knowledge record."""
    chem_name = str(row.get("chem_name", "Unknown chemical"))
    cas = str(row.get("cas_registry_number", "N/A"))
    relevance, keywords = chemical_relevance(row)

    flags: list[str] = []
    if row.get("carc_ind") == "1":
        flags.append("carcinogen")
    if row.get("pfas_ind") == "1":
        flags.append("PFAS")
    if row.get("pbt_ind") == "1":
        flags.append("PBT")

    summary = (
        f"CAS {cas} | EPA TRI regulated chemical"
        + (f" | {', '.join(flags)}" if flags else "")
        + f" | active since {row.get('active_date', 'unknown')}"
    )

    return KnowledgeRecord(
        id=f"epa_tri:{row.get('tri_chem_id', index)}",
        source_type=KnowledgeSourceType.EPA_TRI,
        source_id="tri_chem_info",
        title=chem_name,
        summary=summary[:500],
        category=KnowledgeCategory.CHEMICAL,
        published_at=datetime.now(UTC),
        relevance_score=relevance,
        keywords=keywords,
        metadata={
            "tri_chem_id": row.get("tri_chem_id"),
            "cas_registry_number": cas,
            "active_date": row.get("active_date"),
            "carc_ind": row.get("carc_ind"),
            "caac_ind": row.get("caac_ind"),
            "pfas_ind": row.get("pfas_ind"),
            "pbt_ind": row.get("pbt_ind"),
            "unit_of_measure": row.get("unit_of_measure"),
        },
        url="https://www.epa.gov/toxics-release-inventory-tri-program",
    )


def er_visit_record_from_row(row: dict[str, Any], index: int) -> KnowledgeRecord:
    """Convert a CDC ER visit estimate row into a knowledge record."""
    year = str(row.get("year", "unknown"))
    measure = str(row.get("measure", "Emergency department visits"))
    subgroup = str(row.get("subgroup", "All visits"))
    estimate = row.get("estimate", "N/A")
    relevance, keywords = er_visit_relevance(row)

    try:
        estimate_int = int(str(estimate))
        estimate_label = f"{estimate_int:,} visits"
    except (TypeError, ValueError):
        estimate_label = str(estimate)

    summary = (
        f"{estimate_label} ({subgroup}) in {year} | measure: {row.get('measure_type', 'ED visits')}"
    )
    if row.get("lower_95_ci") and row.get("upper_95_ci"):
        summary += f" | 95% CI {row['lower_95_ci']}-{row['upper_95_ci']}"

    try:
        published = datetime(int(year), 12, 31, tzinfo=UTC)
    except (TypeError, ValueError):
        published = datetime.now(UTC)

    return KnowledgeRecord(
        id=f"cdc_er:{year}:{index}:{subgroup[:20]}",
        source_type=KnowledgeSourceType.CDC_NCHS,
        source_id="cdc_er_visits",
        title=f"{measure} ({year})",
        summary=summary[:500],
        category=KnowledgeCategory.HEALTH,
        published_at=published,
        relevance_score=relevance,
        keywords=keywords,
        metadata={
            "year": year,
            "measure": measure,
            "measure_type": row.get("measure_type"),
            "subgroup": subgroup,
            "group": row.get("group"),
            "estimate": estimate,
            "standard_error": row.get("standard_error"),
            "lower_95_ci": row.get("lower_95_ci"),
            "upper_95_ci": row.get("upper_95_ci"),
            "reliable": row.get("reliable"),
        },
        url="https://data.cdc.gov/NCHS/Estimates-of-Emergency-Department-Visits-in-the-Un/ycxr-emue",
    )
