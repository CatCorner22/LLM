"""Competitor benchmark CLI."""

from __future__ import annotations

import argparse
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.intelligence.benchmark.runner import BenchmarkRunner
from pioneer.intelligence.samples import sample_portfolio

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Pioneer vs competitor baselines")
    parser.parse_args(argv)

    configure_logging()
    portfolio = sample_portfolio()
    report = BenchmarkRunner().run(portfolio)

    print(f"\n=== Pioneer Benchmark: {report.asset_id} ===\n")
    print(report.summary)
    print(f"\nPioneer rank (by risk score): #{report.pioneer_rank}")
    print("\nMetrics vs competitors:")
    for metric in report.metrics:
        win = "WIN" if metric.pioneer_wins else "—"
        print(
            f"  [{win}] {metric.name}: "
            f"Pioneer={metric.pioneer_value} vs avg={metric.competitor_avg} "
            f"({metric.improvement_pct:+.1f}%)"
        )
    print("\nDisruptive advantages:")
    for advantage in report.disruptive_advantages:
        print(f"  • {advantage}")

    logger.info("benchmark_cli_complete", wins=sum(1 for m in report.metrics if m.pioneer_wins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
