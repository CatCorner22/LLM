"""Business advisory CLI."""

from __future__ import annotations

import argparse
import json
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.intelligence.advisory.advisor import BusinessAdvisor
from pioneer.intelligence.samples import sample_portfolio

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate business owner advisory report")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON",
    )
    args = parser.parse_args(argv)

    configure_logging()
    portfolio = sample_portfolio()
    report = BusinessAdvisor().generate_report(portfolio)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print(f"\n=== Pioneer Advisory Report: {report.asset_id} ===\n")
        print(report.executive_summary)
        print(f"\nRisk Score: {report.overall_risk_score:.2f}\n")
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  [{rec.priority.value}] {rec.title}")
            print(f"    {rec.description}")
        if report.scenario_highlights:
            print("\nTop Scenario Impacts:")
            for highlight in report.scenario_highlights:
                print(f"  - {highlight}")

    logger.info("advise_cli_complete", asset_id=report.asset_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
