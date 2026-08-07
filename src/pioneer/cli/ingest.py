"""Ingest public datasets into the Pioneer Intelligence knowledge base."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.intelligence.ingestion.scheduler import (
    IngestionConfig,
    IngestionScheduler,
    KnowledgeStore,
)

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest RSS feeds, EPA chemical inventory, and CDC ER visit data"
    )
    parser.add_argument("--chemical-limit", type=int, default=200)
    parser.add_argument("--er-limit", type=int, default=100)
    parser.add_argument("--min-relevance", type=float, default=0.4)
    parser.add_argument("--feeds-only", action="store_true", help="Skip EPA/CDC datasets")
    parser.add_argument("--public-only", action="store_true", help="Skip RSS feeds")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args(argv)

    configure_logging()
    config = IngestionConfig(
        sources=[] if args.public_only else IngestionConfig().sources,
        chemical_limit=args.chemical_limit,
        er_visit_limit=args.er_limit,
        min_relevance=args.min_relevance,
        ingest_public_data=not args.feeds_only,
    )
    summary = asyncio.run(IngestionScheduler(config).run_once())
    stats = KnowledgeStore().stats()

    if args.json:
        print(json.dumps({"ingestion": summary, "store": stats}, indent=2))
    else:
        print("\n=== Pioneer Knowledge Base Ingestion ===\n")
        print(f"Stored feeds:      {summary['stored_feeds']}")
        print(f"Stored chemicals:  {summary['stored_chemicals']}")
        print(f"Stored ER visits:  {summary['stored_er_visits']}")
        print(f"Pruned records:    {summary['pruned']}")
        print("\nStore totals:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    logger.info("ingest_cli_complete", **summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
