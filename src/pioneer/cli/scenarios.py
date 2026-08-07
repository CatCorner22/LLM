"""Autonomous scenario testing CLI."""

from __future__ import annotations

import argparse
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.intelligence.samples import sample_portfolio
from pioneer.intelligence.scenarios.runner import AutonomousScenarioRunner

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run autonomous risk scenario suite")
    parser.add_argument("--count", type=int, default=6, help="Number of scenarios to run")
    args = parser.parse_args(argv)

    configure_logging()
    portfolio = sample_portfolio()
    result = AutonomousScenarioRunner().run_autonomous_suite(portfolio, count=args.count)

    print(f"\n=== Scenario Analysis: {result.asset_id} ===")
    print(f"Baseline risk: {result.baseline.overall_score:.2f} ({result.baseline.severity.value})")
    print(f"Highest-risk scenario: {result.highest_risk_scenario}\n")

    for outcome in result.outcomes:
        print(
            f"  {outcome.scenario_name}: "
            f"{outcome.baseline_score:.2f} -> {outcome.stressed_score:.2f} "
            f"(delta {outcome.delta:+.2f})"
        )
        print(f"    {outcome.business_impact_summary}")

    logger.info("scenarios_cli_complete", tested=len(result.outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
