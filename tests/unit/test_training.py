"""Unit tests for training orchestrator and SFT config helpers."""

from pathlib import Path

import pytest

from pioneer.core.exceptions import TrainingError
from pioneer.training.callbacks import CheckpointCallback
from pioneer.training.sft import SFTJobConfig, build_sft_args
from pioneer.training.trainer import Trainer, TrainingConfig


@pytest.mark.unit
def test_trainer_run() -> None:
    config = TrainingConfig(experiment_name="test-run", epochs=2, seed=7)

    def train_step(state: dict[str, object]) -> dict[str, float]:
        epoch = int(state.get("epoch", 1))
        return {"loss": 1.0 / epoch, "step": float(epoch)}

    result = Trainer(config=config, train_fn=train_step).run()
    assert result.experiment_name == "test-run"
    assert result.epochs_completed == 2
    assert result.final_loss == 0.5


@pytest.mark.unit
def test_trainer_emits_checkpoints(tmp_path: Path) -> None:
    config = TrainingConfig(experiment_name="ckpt", epochs=2, seed=1)
    callback = CheckpointCallback(checkpoint_dir=tmp_path)

    def train_step(_state: dict[str, object]) -> dict[str, float]:
        return {"loss": 0.5, "step": 1.0}

    result = Trainer(config=config, train_fn=train_step, callbacks=[callback]).run()
    assert len(callback.saved_paths) == 2
    assert result.artifacts
    assert Path(result.artifacts[0]).exists()


@pytest.mark.unit
def test_sft_config_requires_stack_or_builds() -> None:
    config = SFTJobConfig(experiment_name="demo", push_to_hub=False, report_to_trackio=False)
    try:
        import trl  # noqa: F401
    except ImportError:
        with pytest.raises(TrainingError, match=r"pioneer-ml\[training\]"):
            build_sft_args(config)
        return

    args = build_sft_args(config)
    assert getattr(args, "max_length", None) == 1024
