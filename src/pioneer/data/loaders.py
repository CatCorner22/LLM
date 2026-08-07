"""Data loading utilities and factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pioneer.core.exceptions import DataError
from pioneer.core.registry import Registry
from pioneer.data.datasets import DatasetConfig

LoaderFn = Callable[[DatasetConfig], list[dict[str, Any]]]
LOADER_REGISTRY: Registry[LoaderFn] = Registry("data_loader")


def load_jsonl(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    path = config.path
    if not path.exists():
        raise DataError(f"Dataset not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataError(
                    f"Invalid JSON on line {line_number} of {path}",
                    details={"line": line_number},
                ) from exc
    return records


def load_parquet(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load records from a Parquet file."""
    path = config.path
    if not path.exists():
        raise DataError(f"Dataset not found: {path}")
    frame = pd.read_parquet(path)
    records: list[dict[str, Any]] = frame.to_dict(orient="records")
    return records


@LOADER_REGISTRY.register("jsonl")
def _jsonl_loader(config: DatasetConfig) -> list[dict[str, Any]]:
    return load_jsonl(config)


@LOADER_REGISTRY.register("parquet")
def _parquet_loader(config: DatasetConfig) -> list[dict[str, Any]]:
    return load_parquet(config)


class DataLoaderFactory:
    """Resolve and invoke registered data loaders."""

    @staticmethod
    def load(config: DatasetConfig) -> list[dict[str, Any]]:
        loader = LOADER_REGISTRY.get(config.format)
        return loader(config)
