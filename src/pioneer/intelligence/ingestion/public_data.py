"""Ingest public chemical inventory and CDC health datasets into the knowledge base."""

from __future__ import annotations

from typing import Any

import httpx

from pioneer.core.exceptions import IngestionError
from pioneer.core.logging import get_logger
from pioneer.intelligence.ingestion.knowledge import (
    KnowledgeRecord,
    chemical_record_from_row,
    er_visit_record_from_row,
)

logger = get_logger(__name__)

EPA_TRI_CHEM_INFO_URL = "https://data.epa.gov/efservice/TRI_CHEM_INFO/ROWS/{start}:{end}/JSON"
CDC_ER_VISITS_URL = "https://data.cdc.gov/resource/ycxr-emue.json"


class EPAChemicalIngester:
    """Fetch EPA Toxics Release Inventory chemical metadata."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch_chemicals(
        self,
        *,
        start: int = 0,
        limit: int = 200,
        carcinogens_only: bool = False,
    ) -> list[KnowledgeRecord]:
        if carcinogens_only:
            url = (
                "https://data.epa.gov/efservice/TRI_CHEM_INFO/carc_ind/1/"
                f"rows/{start}:{start + limit - 1}/JSON"
            )
        else:
            url = EPA_TRI_CHEM_INFO_URL.format(start=start, end=start + limit - 1)

        rows = await self._fetch_json(url)
        records = [chemical_record_from_row(row, index) for index, row in enumerate(rows)]
        logger.info("epa_chemicals_ingested", count=len(records), carcinogens_only=carcinogens_only)
        return records

    async def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(
                "EPA chemical inventory fetch failed",
                details={"url": url},
            ) from exc

        payload = response.json()
        if not isinstance(payload, list):
            raise IngestionError(
                "Unexpected EPA response format",
                details={"url": url},
            )
        return payload


class CDCEmergencyVisitIngester:
    """Fetch CDC NCHS emergency department visit estimates."""

    DEFAULT_WHERE = (
        "measure_type='By primary diagnosis' AND "
        "(measure like '%Injury%' OR measure like '%poison%' OR leading_10_ranking='0')"
    )

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch_er_visits(
        self,
        *,
        limit: int = 100,
        where: str | None = None,
        years: list[int] | None = None,
    ) -> list[KnowledgeRecord]:
        params: dict[str, str | int] = {
            "$limit": limit,
            "$order": "year DESC",
        }
        filters = [where or self.DEFAULT_WHERE]
        if years:
            year_filter = " OR ".join(f"year='{year}'" for year in years)
            filters.append(f"({year_filter})")
        params["$where"] = " AND ".join(f"({clause})" for clause in filters)

        rows = await self._fetch_json(CDC_ER_VISITS_URL, params)
        records = [er_visit_record_from_row(row, index) for index, row in enumerate(rows)]
        logger.info("cdc_er_visits_ingested", count=len(records))
        return records

    async def _fetch_json(self, url: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, params=params, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(
                "CDC ER visit data fetch failed",
                details={"url": url},
            ) from exc

        payload = response.json()
        if not isinstance(payload, list):
            raise IngestionError(
                "Unexpected CDC response format",
                details={"url": url},
            )
        return payload
