"""Evaluation CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.evaluation.metrics import MetricRegistry

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pioneer ML evaluation CLI")
    parser.add_argument("--metric", default="exact_match", help="Metric to compute")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    args = parser.parse_args(argv)

    configure_logging()
    score = MetricRegistry.compute(args.metric, args.predictions, args.references)
    logger.info("eval_result", metric=args.metric, score=score)
    print(f"{args.metric}: {score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
