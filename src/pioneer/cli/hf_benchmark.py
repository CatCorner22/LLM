"""Hugging Face Hub capability benchmark CLI."""

from __future__ import annotations

import argparse
import json
import sys

from pioneer.core.logging import configure_logging, get_logger
from pioneer.intelligence.benchmark.huggingface import HFBenchmarkConfig, HFBenchmarkRunner

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Pioneer Intelligence on Hugging Face datasets"
    )
    parser.add_argument(
        "--mining-samples",
        type=int,
        default=500,
        help="Number of mining incident rows from HF Hub",
    )
    parser.add_argument(
        "--with-hf-model",
        action="store_true",
        help="Include HF zero-shot model baseline (requires transformers)",
    )
    parser.add_argument(
        "--model",
        default="facebook/bart-large-mnli",
        help="Zero-shot model ID for news relevance task",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args(argv)

    configure_logging()
    config = HFBenchmarkConfig(
        mining_sample_size=args.mining_samples,
        use_hf_zero_shot=args.with_hf_model,
        zero_shot_model=args.model,
    )
    report = HFBenchmarkRunner(config).run()

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print("\n=== Pioneer Hugging Face Benchmark ===\n")
        if report.hf_hub_user:
            print(f"HF Hub user: {report.hf_hub_user}")
        print(report.summary)
        print(f"Elapsed: {report.elapsed_ms:.0f}ms\n")
        for task in report.tasks:
            win = "WIN" if task.pioneer_wins else "—"
            hf_part = ""
            if task.hf_model_score is not None:
                hf_part = f" | HF model={task.hf_model_score}"
            print(
                f"  [{win}] {task.task} ({task.metric_name})\n"
                f"       Dataset: {task.dataset}\n"
                f"       Pioneer={task.pioneer_score} vs baseline={task.baseline_score}"
                f"{hf_part} (n={task.samples})"
            )

    logger.info("hf_benchmark_cli_complete", wins=report.pioneer_task_wins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
