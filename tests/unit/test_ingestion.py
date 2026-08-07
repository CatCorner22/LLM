"""Unit tests for news feed ingestion."""

from datetime import UTC, datetime

import pytest

from pioneer.intelligence.ingestion.feeds import FeedCategory, FeedSource, NewsFeedIngester


RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Storm causes pipe burst and building flood</title>
      <link>https://example.com/1</link>
      <description>Heavy rain led to infrastructure damage and injury reports.</description>
      <pubDate>Fri, 07 Aug 2026 12:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Local sports team wins championship</title>
      <link>https://example.com/2</link>
      <description>Unrelated news item.</description>
      <pubDate>Fri, 07 Aug 2026 11:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.mark.unit
def test_parse_rss_feed() -> None:
    ingester = NewsFeedIngester()
    source = FeedSource(name="test", url="https://example.com/feed", category=FeedCategory.NEWS)
    items = ingester._extract_items(RSS_SAMPLE, source)
    assert len(items) == 2
    assert items[0].keywords
    assert "pipe" in items[0].keywords or "storm" in items[0].keywords


@pytest.mark.unit
def test_filter_relevant() -> None:
    ingester = NewsFeedIngester()
    from pioneer.intelligence.ingestion.feeds import FeedItem

    items = [
        FeedItem(
            id="a",
            title="Low",
            summary="",
            url="",
            source="t",
            category=FeedCategory.NEWS,
            published_at=datetime.now(UTC),
            relevance_score=0.2,
        ),
        FeedItem(
            id="b",
            title="High",
            summary="",
            url="",
            source="t",
            category=FeedCategory.NEWS,
            published_at=datetime.now(UTC),
            relevance_score=0.7,
        ),
    ]
    filtered = ingester.filter_relevant(items, min_score=0.4)
    assert len(filtered) == 1
    assert filtered[0].id == "b"
