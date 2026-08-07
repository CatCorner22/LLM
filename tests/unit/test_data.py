"""Unit tests for data loaders."""

import json
from pathlib import Path

import pytest

from pioneer.core.exceptions import DataError
from pioneer.data.datasets import DatasetConfig, DatasetSplit
from pioneer.data.loaders import DataLoaderFactory, load_jsonl


@pytest.mark.unit
def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    records = [{"text": "hello"}, {"text": "world"}]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    config = DatasetConfig(name="sample", path=path, split=DatasetSplit.TRAIN)
    loaded = load_jsonl(config)
    assert len(loaded) == 2
    assert loaded[0]["text"] == "hello"


@pytest.mark.unit
def test_load_jsonl_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    config = DatasetConfig(name="bad", path=path)
    with pytest.raises(DataError):
        load_jsonl(config)


@pytest.mark.unit
def test_dataloader_factory(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    config = DatasetConfig(name="sample", path=path, format="jsonl")
    records = DataLoaderFactory.load(config)
    assert records == [{"a": 1}]
