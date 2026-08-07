"""Unit tests for knowledge base records and public data ingestion."""

from __future__ import annotations

from typing import Any

import pytest

from pioneer.intelligence.ingestion.knowledge import (
    KnowledgeCategory,
    KnowledgeSourceType,
    chemical_record_from_row,
    chemical_relevance,
    er_visit_record_from_row,
    er_visit_relevance,
)
from pioneer.intelligence.ingestion.scheduler import IngestionConfig, KnowledgeStore


SAMPLE_CHEMICAL: dict[str, Any] = {
    "tri_chem_id": "0000050000",
    "chem_name": "Formaldehyde",
    "active_date": "1987",
    "carc_ind": "1",
    "caac_ind": "1",
    "pfas_ind": "0",
    "pbt_ind": "0",
    "metal_ind": "0",
    "cas_registry_number": "50-00-0",
    "unit_of_measure": "Pounds",
}

SAMPLE_ER_VISIT: dict[str, Any] = {
    "year": "2022",
    "measure_type": "By primary diagnosis",
    "leading_10_ranking": "2",
    "measure": "2. Injury and poisoning",
    "group": "Total",
    "subgroup": "All visits",
    "estimate_type": "Visit count",
    "estimate": "42000000",
    "standard_error": "2500000",
    "lower_95_ci": "37100000",
    "upper_95_ci": "46900000",
    "reliable": "Yes",
}


@pytest.mark.unit
def test_chemical_relevance_scores_carcinogens_high() -> None:
    score, keywords = chemical_relevance(SAMPLE_CHEMICAL)
    assert score >= 0.6
    assert "carcinogen" in keywords


@pytest.mark.unit
def test_er_visit_relevance_scores_injury_high() -> None:
    score, keywords = er_visit_relevance(SAMPLE_ER_VISIT)
    assert score >= 0.6
    assert "injury" in keywords


@pytest.mark.unit
def test_chemical_record_conversion() -> None:
    record = chemical_record_from_row(SAMPLE_CHEMICAL, 0)
    assert record.source_type == KnowledgeSourceType.EPA_TRI
    assert record.category == KnowledgeCategory.CHEMICAL
    assert record.title == "Formaldehyde"
    assert record.metadata["cas_registry_number"] == "50-00-0"


@pytest.mark.unit
def test_er_visit_record_conversion() -> None:
    record = er_visit_record_from_row(SAMPLE_ER_VISIT, 0)
    assert record.source_type == KnowledgeSourceType.CDC_NCHS
    assert record.category == KnowledgeCategory.HEALTH
    assert "2022" in record.title
    assert "42,000,000" in record.summary


@pytest.mark.unit
def test_knowledge_store_append_and_load(tmp_path: Any) -> None:
    store_root = tmp_path / "intelligence"
    store = KnowledgeStore(base_dir=store_root)
    chemical = chemical_record_from_row(SAMPLE_CHEMICAL, 0)
    er_visit = er_visit_record_from_row(SAMPLE_ER_VISIT, 0)

    store.append_knowledge([chemical])
    store.append_knowledge([er_visit])

    chemicals = store.load_knowledge(limit=10, category=KnowledgeCategory.CHEMICAL)
    health = store.load_knowledge(limit=10, category=KnowledgeCategory.HEALTH)
    all_records = store.load_all_knowledge(limit=10)

    assert len(chemicals) == 1
    assert len(health) == 1
    assert len(all_records) == 2
    assert store.stats()["chemical_inventory"] == 1
    assert store.stats()["cdc_er_visits"] == 1


@pytest.mark.unit
def test_ingestion_config_public_data_defaults() -> None:
    config = IngestionConfig()
    assert config.ingest_public_data is True
    assert config.chemical_limit == 200
    assert config.er_visit_limit == 100
