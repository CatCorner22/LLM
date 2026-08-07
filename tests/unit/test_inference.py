"""Unit tests for batch inference engine."""

import pytest

from pioneer.core.exceptions import InferenceError
from pioneer.inference.engine import BatchInferenceEngine, InferenceConfig


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_inference_success() -> None:
    def predict(payload: str) -> str:
        return payload.upper()

    engine = BatchInferenceEngine(predict_fn=predict, config=InferenceConfig(max_concurrency=2))
    results = await engine.run([("a", "hello"), ("b", "world")])
    assert len(results) == 2
    assert results[0].output == "HELLO"
    assert results[1].output == "WORLD"
    assert all(result.latency_ms >= 0 for result in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_inference_failure() -> None:
    def predict(_payload: str) -> str:
        raise RuntimeError("boom")

    engine = BatchInferenceEngine(predict_fn=predict)
    with pytest.raises(InferenceError):
        await engine.run([("fail", "x")])
