"""Unit tests for training orchestrator."""

import pytest

from pioneer.training.trainer import Trainer, TrainingConfig


@pytest.mark.unit
def test_trainer_run() -> None:
    config = TrainingConfig(experiment_name="test-run", epochs=2, seed=7)

    def train_step(state: dict[str, object]) -> dict[str, float]:
        epoch = int(state.get("epoch", 1))
        return {"loss": 1.0 / epoch}

    result = Trainer(config=config, train_fn=train_step).run()
    assert result.experiment_name == "test-run"
    assert result.epochs_completed == 2
    assert result.final_loss == 0.5
