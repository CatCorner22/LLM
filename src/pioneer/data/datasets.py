"""Dataset configuration and base dataset types."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from pioneer.core.exceptions import DataError


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetConfig(BaseModel):
    """Declarative dataset configuration."""

    name: str
    path: Path
    split: DatasetSplit = DatasetSplit.TRAIN
    format: str = "jsonl"
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = 42
    columns: dict[str, str] = Field(default_factory=dict)


class PioneerDataset:
    """Lazy, validated dataset wrapper."""

    def __init__(self, config: DatasetConfig, records: list[dict[str, Any]] | None = None) -> None:
        self.config = config
        self._records = records

    @property
    def records(self) -> list[dict[str, Any]]:
        if self._records is None:
            raise DataError("Dataset not loaded; call load() first")
        return self._records

    def load(self, loader: Any) -> PioneerDataset:
        """Load records using a provided loader callable."""
        raw = loader(self.config)
        if self.config.max_samples is not None:
            raw = raw[: self.config.max_samples]
        self._records = raw
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.records

    def validate_schema(self, required_fields: set[str]) -> None:
        for index, record in enumerate(self.records):
            missing = required_fields - set(record)
            if missing:
                raise DataError(
                    f"Record {index} missing fields: {sorted(missing)}",
                    details={"index": index, "missing": sorted(missing)},
                )
