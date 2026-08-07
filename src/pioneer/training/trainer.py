"""Generic training orchestrator with callback support."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pioneer.core.config import get_settings
from pioneer.core.exceptions import TrainingError
from pioneer.core.logging import configure_logging, get_logger
from pioneer.training.callbacks import Callback, LoggingCallback

logger = get_logger(__name__)


class TrainingConfig(BaseModel):
    """Training hyperparameters and runtime configuration."""

    experiment_name: str
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    seed: int = 42
    log_every_n_steps: int = Field(default=10, ge=1)
    max_steps: int | None = Field(default=None, ge=1)


@dataclass
class TrainingResult:
    """Outcome of a training run."""

    experiment_name: str
    epochs_completed: int
    final_loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


class Trainer:
    """Orchestrate training with reproducibility and callbacks."""

    def __init__(
        self,
        config: TrainingConfig,
        train_fn: Callable[[dict[str, Any]], dict[str, float]],
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.config = config
        self.train_fn = train_fn
        self.callbacks = callbacks or [LoggingCallback()]
        self._state: dict[str, Any] = {}

    def _set_seed(self) -> None:
        settings = get_settings()
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        settings.seed = seed

    def _emit(self, hook: str) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(self._state)

    def run(self) -> TrainingResult:
        configure_logging()
        self._set_seed()

        self._state = {
            "experiment_name": self.config.experiment_name,
            "epoch": 0,
            "step": 0,
            "loss": float("inf"),
        }
        self._emit("on_train_begin")

        final_metrics: dict[str, float] = {}
        try:
            for epoch in range(1, self.config.epochs + 1):
                self._state["epoch"] = epoch
                self._emit("on_epoch_begin")

                metrics = self.train_fn(self._state)
                final_metrics = metrics
                self._state.update(metrics)
                self._emit("on_epoch_end")

                if self.config.max_steps and self._state.get("step", 0) >= self.config.max_steps:
                    break
        except Exception as exc:
            raise TrainingError(
                f"Training failed for experiment '{self.config.experiment_name}'",
                details={"epoch": self._state.get("epoch")},
            ) from exc

        self._emit("on_train_end")
        logger.info("training_result", metrics=final_metrics)

        return TrainingResult(
            experiment_name=self.config.experiment_name,
            epochs_completed=self._state["epoch"],
            final_loss=final_metrics.get("loss", float("inf")),
            metrics=final_metrics,
        )
