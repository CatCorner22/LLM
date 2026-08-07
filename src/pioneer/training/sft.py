"""TRL supervised fine-tuning (SFT) for Pioneer ML.

Uses modern TRL APIs:
- ``SFTConfig(max_length=...)`` (not legacy ``max_seq_length``)
- LoRA via PEFT
- Trackio reporting when available
- Hub push (critical — Jobs environments are ephemeral)
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from pioneer.core.exceptions import TrainingError
from pioneer.core.logging import get_logger
from pioneer.training.trainer import TrainingResult

logger = get_logger(__name__)


class SFTJobConfig(BaseModel):
    """Configuration for a TRL SFT run."""

    experiment_name: str
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "trl-lib/Capybara"
    dataset_split: str = "train"
    hub_model_id: str | None = None
    output_dir: str = "artifacts/sft"
    num_train_epochs: int = Field(default=1, ge=1)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_length: int = Field(default=1024, ge=64)
    max_steps: int | None = Field(default=None, ge=1)
    eval_fraction: float = Field(default=0.1, ge=0.0, lt=0.5)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    seed: int = 42
    push_to_hub: bool = True
    report_to_trackio: bool = True
    project: str = "pioneer-ml"
    run_name: str | None = None


def _require_training_stack() -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError(
            "TRL training stack not installed. Run: pip install 'pioneer-ml[training]'",
            details={"missing": str(exc)},
        ) from exc
    return {
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def build_sft_args(config: SFTJobConfig) -> Any:
    """Build a TRL ``SFTConfig`` with Hub push and Trackio defaults."""
    stack = _require_training_stack()
    hub_model_id = config.hub_model_id
    if config.push_to_hub and not hub_model_id:
        username = os.environ.get("HF_USERNAME") or os.environ.get("HF_HUB_USER")
        if username:
            hub_model_id = f"{username}/{config.experiment_name}"
        else:
            raise TrainingError(
                "push_to_hub requires hub_model_id or HF_USERNAME environment variable"
            )

    report_to: str | list[str] = "none"
    if config.report_to_trackio:
        try:
            import trackio  # noqa: F401

            report_to = "trackio"
        except ImportError:
            logger.warning("trackio_unavailable", hint="pip install trackio")
            report_to = "none"

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "seed": config.seed,
        "report_to": report_to,
        "run_name": config.run_name or config.experiment_name,
        "push_to_hub": config.push_to_hub,
        "hub_model_id": hub_model_id,
        "hub_strategy": "every_save" if config.push_to_hub else "end",
        "gradient_checkpointing": True,
    }
    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps
    if config.eval_fraction > 0:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = 50
    # Trackio project name when supported by installed TRL
    if report_to == "trackio":
        kwargs["project"] = config.project

    try:
        return stack["SFTConfig"](**kwargs)
    except TypeError:
        kwargs.pop("project", None)
        return stack["SFTConfig"](**kwargs)


def run_sft(config: SFTJobConfig) -> TrainingResult:
    """Execute a TRL SFT job and optionally push the adapter to the Hub."""
    stack = _require_training_stack()
    load_dataset = stack["load_dataset"]
    lora_cls = stack["LoraConfig"]
    trainer_cls = stack["SFTTrainer"]

    if config.push_to_hub and not os.environ.get("HF_TOKEN"):
        logger.warning(
            "hf_token_missing",
            hint="Set HF_TOKEN or pass secrets when using HF Jobs — results are ephemeral",
        )

    logger.info(
        "sft_loading_dataset",
        dataset=config.dataset_name,
        split=config.dataset_split,
    )
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    eval_dataset = None
    train_dataset = dataset
    if config.eval_fraction > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=config.eval_fraction, seed=config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    peft_config = lora_cls(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    args = build_sft_args(config)
    trainer = trainer_cls(
        model=config.model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        peft_config=peft_config,
    )

    logger.info("sft_train_begin", model=config.model_name, samples=len(train_dataset))
    train_output = trainer.train()
    metrics = {
        k: float(v) for k, v in (train_output.metrics or {}).items() if isinstance(v, (int, float))
    }

    artifacts: list[str] = [config.output_dir]
    if config.push_to_hub:
        logger.info("sft_push_to_hub", hub_model_id=config.hub_model_id or args.hub_model_id)
        trainer.push_to_hub()
        if config.hub_model_id or getattr(args, "hub_model_id", None):
            artifacts.append(f"hub://{config.hub_model_id or args.hub_model_id}")

    try:
        import trackio

        trackio.finish()
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - optional monitoring cleanup
        logger.warning("trackio_finish_failed", error=str(exc))

    return TrainingResult(
        experiment_name=config.experiment_name,
        epochs_completed=int(metrics.get("epoch", config.num_train_epochs)),
        final_loss=float(metrics.get("train_loss", metrics.get("loss", float("inf")))),
        metrics=metrics,
        artifacts=artifacts,
    )
