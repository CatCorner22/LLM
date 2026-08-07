"""Training CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from pioneer.core.config import get_settings
from pioneer.core.logging import configure_logging, get_logger
from pioneer.training.trainer import Trainer, TrainingConfig

logger = get_logger(__name__)


def _dummy_train_step(state: dict[str, object]) -> dict[str, float]:
    epoch_value = state.get("epoch", 1)
    epoch = epoch_value if isinstance(epoch_value, int) else 1
    return {"loss": max(0.01, 1.0 / epoch), "step": float(epoch)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pioneer ML training CLI")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    logger.info("train_cli_start", experiment=args.experiment, env=settings.env)

    config = TrainingConfig(
        experiment_name=args.experiment,
        epochs=args.epochs,
        seed=args.seed,
    )
    trainer = Trainer(config=config, train_fn=_dummy_train_step)
    result = trainer.run()

    logger.info(
        "train_cli_complete",
        experiment=result.experiment_name,
        final_loss=result.final_loss,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
