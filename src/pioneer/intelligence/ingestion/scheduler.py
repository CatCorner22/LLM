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


class KnowledgeStore:
    """Persist ingested intelligence for advisory and scenario engines."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or settings.data_dir / "intelligence" / "feed_items.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, items: list[FeedItem]) -> int:
        with self.path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(item.model_dump_json() + "\n")
        return len(items)

    def load_recent(self, limit: int = 100) -> list[FeedItem]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        items = [FeedItem.model_validate(json.loads(line)) for line in lines[-limit:]]
        return list(reversed(items))

    def prune(self, retention_days: int) -> int:
        if not self.path.exists():
            return 0
        cutoff = datetime.now(UTC).timestamp() - retention_days * 86400
        kept: list[FeedItem] = []
        removed = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = FeedItem.model_validate(json.loads(line))
            if item.published_at.timestamp() >= cutoff:
                kept.append(item)
            else:
                removed += 1
        with self.path.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(item.model_dump_json() + "\n")
        return removed


class IngestionScheduler:
    """Run periodic feed ingestion and persist to knowledge store."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        ingester: NewsFeedIngester | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.ingester = ingester or NewsFeedIngester()
        self.store = KnowledgeStore(self.config.store_path)

    async def run_once(self) -> list[FeedItem]:
        raw_items = await self.ingester.fetch_all(self.config.sources)
        relevant = self.ingester.filter_relevant(raw_items, self.config.min_relevance)
        self.store.append(relevant)
        self.store.prune(self.config.retention_days)
        logger.info(
            "ingestion_complete",
            fetched=len(raw_items),
            stored=len(relevant),
        )
        return relevant
