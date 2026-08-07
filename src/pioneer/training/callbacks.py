"""Training callbacks for observability and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pioneer.core.logging import get_logger

logger = get_logger(__name__)


class Callback:
    """Hook into training lifecycle events."""

    def on_train_begin(self, state: dict[str, Any]) -> None:
        pass

    def on_epoch_begin(self, state: dict[str, Any]) -> None:
        pass

    def on_step_end(self, state: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        pass

    def on_train_end(self, state: dict[str, Any]) -> None:
        pass


class LoggingCallback(Callback):
    """Structured logging for training progress."""

    def on_train_begin(self, state: dict[str, Any]) -> None:
        logger.info("training_started", **state)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        logger.info("epoch_completed", **state)

    def on_train_end(self, state: dict[str, Any]) -> None:
        logger.info("training_completed", **state)


class CheckpointCallback(Callback):
    """Persist training checkpoints on interval."""

    def __init__(self, checkpoint_dir: Path, save_every_n_epochs: int = 1) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state.get("epoch", 0)
        if epoch % self.save_every_n_epochs != 0:
            return
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.json"
        path.write_text(__import__("json").dumps(state, default=str, indent=2), encoding="utf-8")
        logger.info("checkpoint_saved", path=str(path), epoch=epoch)
