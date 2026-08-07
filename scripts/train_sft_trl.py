#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.12.0",
#     "transformers>=4.44.0",
#     "accelerate>=0.33.0",
#     "datasets>=2.20.0",
#     "trackio",
#     "huggingface-hub>=0.24",
# ]
# ///
"""UV script for TRL SFT on Hugging Face Jobs.

Environment variables:
  HUB_MODEL_ID   (required) destination repo for adapters
  MODEL_NAME     base model (default Qwen/Qwen2.5-0.5B)
  DATASET_NAME   dataset (default trl-lib/Capybara)
  EXPERIMENT_NAME Trackio/run name
  MAX_STEPS      optional step cap for demos
  HF_TOKEN       provided via Jobs secrets
"""

from __future__ import annotations

import os
import sys

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main() -> int:
    hub_model_id = os.environ.get("HUB_MODEL_ID")
    if not hub_model_id:
        print("ERROR: HUB_MODEL_ID is required — Jobs are ephemeral", file=sys.stderr)
        return 1
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN missing from job secrets", file=sys.stderr)
        return 1

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    dataset_name = os.environ.get("DATASET_NAME", "trl-lib/Capybara")
    experiment = os.environ.get("EXPERIMENT_NAME", "pioneer-sft")
    max_steps_raw = os.environ.get("MAX_STEPS")
    max_steps = int(max_steps_raw) if max_steps_raw else None

    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    # Demo-friendly: skip eval to save memory on smaller GPUs
    train_dataset = dataset.select(range(min(len(dataset), 500)))

    config_kwargs = {
        "output_dir": experiment,
        "push_to_hub": True,
        "hub_model_id": hub_model_id,
        "hub_strategy": "every_save",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "max_length": 512,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 1,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "report_to": "trackio",
        "run_name": experiment,
        "project": "pioneer-ml",
    }
    if max_steps is not None:
        config_kwargs["max_steps"] = max_steps

    try:
        args = SFTConfig(**config_kwargs)
    except TypeError:
        config_kwargs.pop("project", None)
        args = SFTConfig(**config_kwargs)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    trainer = SFTTrainer(
        model=model_name,
        train_dataset=train_dataset,
        args=args,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.push_to_hub()
    trackio.finish()
    print(f"Pushed to https://huggingface.co/{hub_model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
