"""Scheduled ingestion and knowledge retention."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pioneer.core.config import get_settings
from pioneer.core.logging import get_logger
from pioneer.intelligence.ingestion.feeds import (
    FeedCategory,
    FeedItem,
    FeedSource,
    NewsFeedIngester,
)
from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory, KnowledgeRecord
from pioneer.intelligence.ingestion.public_data import (
    CDCEmergencyVisitIngester,
    EPAChemicalIngester,
)

logger = get_logger(__name__)

DEFAULT_SOURCES: list[FeedSource] = [
    FeedSource(
        name="reuters_world",
        url="https://feeds.reuters.com/reuters/worldNews",
        category=FeedCategory.NEWS,
        keywords=["infrastructure", "disaster", "flood"],
    ),
    FeedSource(
        name="bbc_world",
        url="http://feeds.bbci.co.uk/news/world/rss.xml",
        category=FeedCategory.NEWS,
        keywords=["storm", "weather", "building"],
    ),
]


class IngestionConfig(BaseModel):
    sources: list[FeedSource] = Field(default_factory=lambda: list(DEFAULT_SOURCES))
    min_relevance: float = Field(default=0.4, ge=0.0, le=1.0)
    retention_days: int = Field(default=30, ge=1)
    store_path: Path | None = None
    chemical_limit: int = Field(default=200, ge=1)
    er_visit_limit: int = Field(default=100, ge=1)
    ingest_public_data: bool = True


class KnowledgeStore:
    """Persist ingested intelligence for advisory and scenario engines."""

    def __init__(self, path: Path | None = None, base_dir: Path | None = None) -> None:
        settings = get_settings()
        root = base_dir or settings.data_dir / "intelligence"
        self.path = path or root / "feed_items.jsonl"
        self.knowledge_path = root / "knowledge_base.jsonl"
        self.chemical_path = root / "chemical_inventory.jsonl"
        self.health_path = root / "cdc_er_visits.jsonl"
        for store_path in (self.path, self.knowledge_path, self.chemical_path, self.health_path):
            store_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, items: list[FeedItem]) -> int:
        with self.path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(item.model_dump_json() + "\n")
        return len(items)

    def append_knowledge(
        self, records: list[KnowledgeRecord], *, category: str | None = None
    ) -> int:
        target = self._path_for_category(category, records)
        with target.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        with self.knowledge_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        return len(records)

    def _path_for_category(
        self, category: str | None, records: list[KnowledgeRecord]
    ) -> Path:
        if category == "chemical" or (
            records and records[0].category == KnowledgeCategory.CHEMICAL
        ):
            return self.chemical_path
        if category == "health" or (
            records and records[0].category == KnowledgeCategory.HEALTH
        ):
            return self.health_path
        return self.knowledge_path

    def load_recent(self, limit: int = 100) -> list[FeedItem]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        items = [FeedItem.model_validate(json.loads(line)) for line in lines[-limit:]]
        return list(reversed(items))

    def load_knowledge(
        self,
        limit: int = 100,
        *,
        category: KnowledgeCategory | None = None,
    ) -> list[KnowledgeRecord]:
        if category == KnowledgeCategory.CHEMICAL and self.chemical_path.exists():
            source_path = self.chemical_path
        elif category == KnowledgeCategory.HEALTH and self.health_path.exists():
            source_path = self.health_path
        elif self.knowledge_path.exists():
            source_path = self.knowledge_path
        else:
            return []

        lines = source_path.read_text(encoding="utf-8").strip().splitlines()
        records = [
            KnowledgeRecord.model_validate(json.loads(line)) for line in lines[-limit * 2 :]
        ]
        if category is not None:
            records = [record for record in records if record.category == category]
        records.sort(key=lambda record: record.relevance_score, reverse=True)
        return records[:limit]

    def load_all_knowledge(self, limit: int = 100) -> list[KnowledgeRecord]:
        combined: list[KnowledgeRecord] = []
        for path in (self.knowledge_path, self.chemical_path, self.health_path):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                combined.append(KnowledgeRecord.model_validate(json.loads(line)))
        combined.sort(
            key=lambda record: (record.relevance_score, record.published_at.timestamp()),
            reverse=True,
        )
        seen: set[str] = set()
        deduped: list[KnowledgeRecord] = []
        for record in combined:
            if record.id in seen:
                continue
            seen.add(record.id)
            deduped.append(record)
        return deduped[:limit]

    def prune(self, retention_days: int) -> int:
        removed = self._prune_file(self.path, retention_days)
        removed += self._prune_file(self.knowledge_path, retention_days)
        return removed

    def _prune_file(self, path: Path, retention_days: int) -> int:
        if not path.exists():
            return 0
        cutoff = datetime.now(UTC).timestamp() - retention_days * 86400
        kept_lines: list[str] = []
        removed = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            published_raw = payload.get("published_at")
            if not published_raw:
                kept_lines.append(line)
                continue
            published = datetime.fromisoformat(str(published_raw).replace("Z", "+00:00"))
            if published.timestamp() >= cutoff:
                kept_lines.append(line)
            else:
                removed += 1
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
        return removed

    def stats(self) -> dict[str, int]:
        return {
            "feed_items": self._count_lines(self.path),
            "knowledge_records": self._count_lines(self.knowledge_path),
            "chemical_inventory": self._count_lines(self.chemical_path),
            "cdc_er_visits": self._count_lines(self.health_path),
        }

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class IngestionScheduler:
    """Run periodic feed ingestion and persist to knowledge store."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        ingester: NewsFeedIngester | None = None,
        chemical_ingester: EPAChemicalIngester | None = None,
        er_ingester: CDCEmergencyVisitIngester | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.ingester = ingester or NewsFeedIngester()
        self.chemical_ingester = chemical_ingester or EPAChemicalIngester()
        self.er_ingester = er_ingester or CDCEmergencyVisitIngester()
        store_root = None
        if self.config.store_path is not None:
            store_root = self.config.store_path.parent
        self.store = KnowledgeStore(self.config.store_path, base_dir=store_root)

    async def run_once(self) -> dict[str, int]:
        raw_items = await self.ingester.fetch_all(self.config.sources)
        relevant = self.ingester.filter_relevant(raw_items, self.config.min_relevance)
        stored_feeds = self.store.append(relevant)

        stored_chemicals = 0
        stored_er = 0
        if self.config.ingest_public_data:
            chemicals = await self.chemical_ingester.fetch_chemicals(
                limit=self.config.chemical_limit,
                carcinogens_only=True,
            )
            relevant_chemicals = [
                record
                for record in chemicals
                if record.relevance_score >= self.config.min_relevance
            ]
            stored_chemicals = self.store.append_knowledge(relevant_chemicals)

            er_visits = await self.er_ingester.fetch_er_visits(limit=self.config.er_visit_limit)
            relevant_er = [
                record
                for record in er_visits
                if record.relevance_score >= self.config.min_relevance
            ]
            stored_er = self.store.append_knowledge(relevant_er)

        removed = self.store.prune(self.config.retention_days)
        logger.info(
            "ingestion_complete",
            fetched=len(raw_items),
            stored_feeds=stored_feeds,
            stored_chemicals=stored_chemicals,
            stored_er_visits=stored_er,
            pruned=removed,
        )
        return {
            "fetched_feeds": len(raw_items),
            "stored_feeds": stored_feeds,
            "stored_chemicals": stored_chemicals,
            "stored_er_visits": stored_er,
            "pruned": removed,
        }
