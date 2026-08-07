"""External information ingestion for news and regulatory feeds."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field

from pioneer.core.exceptions import IngestionError
from pioneer.core.logging import get_logger

logger = get_logger(__name__)


class FeedCategory(StrEnum):
    NEWS = "news"
    WEATHER = "weather"
    REGULATORY = "regulatory"
    INDUSTRY = "industry"


class FeedItem(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    source: str
    category: FeedCategory
    published_at: datetime
    keywords: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class FeedSource(BaseModel):
    name: str
    url: str
    category: FeedCategory = FeedCategory.NEWS
    keywords: list[str] = Field(default_factory=list)


class NewsFeedIngester:
    """Ingest RSS/Atom feeds and score relevance to risk domains."""

    RISK_KEYWORDS = frozenset(
        {
            "flood",
            "storm",
            "hurricane",
            "freeze",
            "pipe",
            "burst",
            "leak",
            "building",
            "collapse",
            "injury",
            "accident",
            "osha",
            "infrastructure",
            "water",
            "sewer",
            "inspection",
            "recall",
            "fire",
            "chemical",
            "toxic",
            "poison",
            "carcinogen",
            "hazmat",
            "spill",
            "exposure",
        }
    )

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def _parse_datetime(self, raw: str | None) -> datetime:
        if not raw:
            return datetime.now(UTC)
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:
                parsed = datetime.strptime(raw.strip(), fmt)
                return parsed.astimezone(UTC)
            except ValueError:
                continue
        return datetime.now(UTC)

    def _extract_items(self, content: str, source: FeedSource) -> list[FeedItem]:
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            raise IngestionError(
                f"Failed to parse feed: {source.url}",
                details={"source": source.name},
            ) from exc

        items: list[FeedItem] = []
        for index, item in enumerate(root.findall(".//item") + root.findall(".//{*}entry")):
            title_el = item.find("title")
            if title_el is None:
                title_el = item.find("{*}title")
            link_el = item.find("link")
            if link_el is None:
                link_el = item.find("{*}link")
            desc_el = item.find("description")
            if desc_el is None:
                desc_el = item.find("summary")
            if desc_el is None:
                desc_el = item.find("{*}summary")
            if desc_el is None:
                desc_el = item.find("{*}content")
            date_el = item.find("pubDate")
            if date_el is None:
                date_el = item.find("published")
            if date_el is None:
                date_el = item.find("{*}published")
            if date_el is None:
                date_el = item.find("{*}updated")

            title = (title_el.text or "").strip() if title_el is not None else "Untitled"
            summary = (desc_el.text or "").strip() if desc_el is not None else ""
            url = ""
            if link_el is not None:
                url = link_el.text or link_el.get("href") or ""
            published = self._parse_datetime(date_el.text if date_el is not None else None)

            text_blob = f"{title} {summary}".lower()
            matched = [kw for kw in self.RISK_KEYWORDS if kw in text_blob]
            source_matches = [kw for kw in source.keywords if kw in text_blob]
            all_keywords = sorted(set(matched + source_matches))
            relevance = min(1.0, 0.3 + 0.1 * len(all_keywords))

            items.append(
                FeedItem(
                    id=f"{source.name}:{index}:{published.timestamp()}",
                    title=title,
                    summary=summary[:500],
                    url=url,
                    source=source.name,
                    category=source.category,
                    published_at=published,
                    keywords=all_keywords,
                    relevance_score=relevance,
                )
            )
        return items

    async def fetch_feed(self, source: FeedSource) -> list[FeedItem]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(source.url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(
                f"Feed fetch failed: {source.name}",
                details={"url": source.url},
            ) from exc

        items = self._extract_items(response.text, source)
        logger.info("feed_ingested", source=source.name, count=len(items))
        return items

    async def fetch_all(self, sources: list[FeedSource]) -> list[FeedItem]:
        all_items: list[FeedItem] = []
        for source in sources:
            all_items.extend(await self.fetch_feed(source))
        all_items.sort(key=lambda item: item.published_at, reverse=True)
        return all_items

    def filter_relevant(self, items: list[FeedItem], min_score: float = 0.4) -> list[FeedItem]:
        return [item for item in items if item.relevance_score >= min_score]
