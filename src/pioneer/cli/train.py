"""Training CLI entrypoint — stub orchestrator or TRL SFT backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pioneer.core.config import get_settings
from pioneer.core.logging import configure_logging, get_logger
from pioneer.training.callbacks import CheckpointCallback, LoggingCallback
from pioneer.training.trainer import Trainer, TrainingConfig

logger = get_logger(__name__)


def _dummy_train_step(state: dict[str, object]) -> dict[str, float]:
    epoch_value = state.get("epoch", 1)
    epoch = epoch_value if isinstance(epoch_value, int) else 1
    step_value = state.get("step", 0)
    step = float(step_value) if isinstance(step_value, (int, float)) else 0.0
    return {"loss": max(0.01, 1.0 / epoch), "step": step + 1.0}


def _run_stub(args: argparse.Namespace) -> int:
    config = TrainingConfig(
        experiment_name=args.experiment,
        epochs=args.epochs,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    callbacks = [
        LoggingCallback(),
        CheckpointCallback(checkpoint_dir=checkpoint_dir),
    ]
    trainer = Trainer(config=config, train_fn=_dummy_train_step, callbacks=callbacks)
    result = trainer.run()
    logger.info(
        "train_cli_complete",
        backend="stub",
        experiment=result.experiment_name,
        final_loss=result.final_loss,
        artifacts=result.artifacts,
    )
    return 0


def _run_sft(args: argparse.Namespace) -> int:
    from pioneer.training.sft import SFTJobConfig, run_sft

    config = SFTJobConfig(
        experiment_name=args.experiment,
        model_name=args.model,
        dataset_name=args.dataset,
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        report_to_trackio=not args.no_trackio,
        project=args.project,
        run_name=args.run_name or args.experiment,
        eval_fraction=0.0 if args.no_eval else 0.1,
    )
    result = run_sft(config)
    logger.info(
        "train_cli_complete",
        backend="sft",
        experiment=result.experiment_name,
        final_loss=result.final_loss,
        artifacts=result.artifacts,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pioneer ML training CLI")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument(
        "--backend",
        choices=("stub", "sft"),
        default="stub",
        help="stub = lightweight orchestrator; sft = TRL supervised fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps")
    parser.add_argument(
        "--output-dir",
        default="artifacts/training",
        help="Local output / checkpoint directory",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base model for SFT")
    parser.add_argument("--dataset", default="trl-lib/Capybara", help="HF dataset for SFT")
    parser.add_argument("--hub-model-id", default=None, help="Hub repo id for push (user/model)")
    parser.add_argument("--max-length", type=int, default=1024, help="SFT max sequence length")
    parser.add_argument("--project", default="pioneer-ml", help="Trackio project name")
    parser.add_argument("--run-name", default=None, help="Trackio run name")
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push SFT adapters to the Hub (required for HF Jobs)",
    )
    parser.add_argument("--no-trackio", action="store_true", help="Disable Trackio reporting")
    parser.add_argument("--no-eval", action="store_true", help="Skip eval split (saves memory)")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    logger.info(
        "train_cli_start",
        experiment=args.experiment,
        backend=args.backend,
        env=settings.env,
    )

    if args.backend == "sft":
        return _run_sft(args)
    return _run_stub(args)


if __name__ == "__main__":
    sys.exit(main())
