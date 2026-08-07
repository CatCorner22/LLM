"""Integration tests for public dataset ingestion."""

from __future__ import annotations

from typing import Any

import pytest

from pioneer.intelligence.ingestion.knowledge import (
    KnowledgeCategory,
    chemical_record_from_row,
    er_visit_record_from_row,
)
from pioneer.intelligence.ingestion.public_data import (
    CDCEmergencyVisitIngester,
    EPAChemicalIngester,
)
from pioneer.intelligence.ingestion.scheduler import IngestionConfig, IngestionScheduler

EPA_SAMPLE = [
    {
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
]

CDC_SAMPLE = [
    {
        "year": "2022",
        "measure_type": "By primary diagnosis",
        "leading_10_ranking": "2",
        "measure": "2. Injury and poisoning",
        "group": "Total",
        "subgroup": "All visits",
        "estimate": "42000000",
        "lower_95_ci": "37100000",
        "upper_95_ci": "46900000",
    }
]


class MockEPAChemicalIngester(EPAChemicalIngester):
    async def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        return EPA_SAMPLE


class MockCDCEmergencyVisitIngester(CDCEmergencyVisitIngester):
    async def _fetch_json(self, url: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
        return CDC_SAMPLE


@pytest.mark.asyncio
async def test_epa_chemical_ingester_returns_records() -> None:
    records = await MockEPAChemicalIngester().fetch_chemicals(limit=1)
    assert len(records) == 1
    assert records[0].category == KnowledgeCategory.CHEMICAL
    assert records[0].title == "Formaldehyde"


@pytest.mark.asyncio
async def test_cdc_er_ingester_returns_records() -> None:
    records = await MockCDCEmergencyVisitIngester().fetch_er_visits(limit=1)
    assert len(records) == 1
    assert records[0].category == KnowledgeCategory.HEALTH
    assert "Injury" in records[0].title


@pytest.mark.asyncio
async def test_scheduler_stores_public_data(tmp_path: Any) -> None:
    store_root = tmp_path / "intelligence"
    config = IngestionConfig(
        sources=[],
        store_path=store_root / "feed_items.jsonl",
        chemical_limit=1,
        er_visit_limit=1,
    )
    scheduler = IngestionScheduler(
        config,
        chemical_ingester=MockEPAChemicalIngester(),
        er_ingester=MockCDCEmergencyVisitIngester(),
    )
    summary = await scheduler.run_once()
    assert summary["stored_chemicals"] == 1
    assert summary["stored_er_visits"] == 1

    chemicals = scheduler.store.load_knowledge(10, category=KnowledgeCategory.CHEMICAL)
    health = scheduler.store.load_knowledge(10, category=KnowledgeCategory.HEALTH)
    assert len(chemicals) == 1
    assert len(health) == 1


@pytest.mark.unit
def test_record_helpers() -> None:
    chemical = chemical_record_from_row(EPA_SAMPLE[0], 0)
    er_visit = er_visit_record_from_row(CDC_SAMPLE[0], 0)
    assert chemical.metadata["carc_ind"] == "1"
    assert er_visit.metadata["year"] == "2022"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_public_data_ingestion_live() -> None:
    config = IngestionConfig(
        sources=[],
        chemical_limit=10,
        er_visit_limit=10,
        ingest_public_data=True,
    )
    summary = await IngestionScheduler(config).run_once()
    assert summary["stored_chemicals"] >= 1
    assert summary["stored_er_visits"] >= 1
