"""Training callbacks for observability and checkpointing."""

from __future__ import annotations

import json
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

    def on_step_end(self, state: dict[str, Any]) -> None:
        step = int(state.get("step", 0))
        every = int(state.get("log_every_n_steps", 10) or 10)
        if step % every == 0:
            logger.info(
                "step_completed",
                epoch=state.get("epoch"),
                step=state.get("step"),
                loss=state.get("loss"),
            )

    def on_train_end(self, state: dict[str, Any]) -> None:
        logger.info("training_completed", **state)


class CheckpointCallback(Callback):
    """Persist training checkpoints on interval."""

    def __init__(self, checkpoint_dir: Path, save_every_n_epochs: int = 1) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_paths: list[str] = []

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = int(state.get("epoch", 0))
        if epoch == 0 or epoch % self.save_every_n_epochs != 0:
            return
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.json"
        payload = {key: value for key, value in state.items() if key != "artifacts"}
        path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        self.saved_paths.append(str(path))
        artifacts = state.setdefault("artifacts", [])
        if isinstance(artifacts, list):
            artifacts.append(str(path))
        logger.info("checkpoint_saved", path=str(path), epoch=epoch)
