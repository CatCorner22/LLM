"""Batch and streaming inference engine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from pioneer.core.exceptions import InferenceError
from pioneer.core.logging import get_logger

logger = get_logger(__name__)


class InferenceConfig(BaseModel):
    """Inference runtime configuration."""

    batch_size: int = Field(default=8, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    retry_attempts: int = Field(default=2, ge=0)


@dataclass
class InferenceResult:
    """Single inference output."""

    input_id: str
    output: Any
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchInferenceEngine:
    """Concurrent batch inference with backpressure."""

    def __init__(
        self,
        predict_fn: Callable[[Any], Any],
        config: InferenceConfig | None = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.config = config or InferenceConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def _predict_one(self, input_id: str, payload: Any) -> InferenceResult:
        async with self._semaphore:
            start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(self.predict_fn):
                    output = await asyncio.wait_for(
                        self.predict_fn(payload),
                        timeout=self.config.timeout_seconds,
                    )
                else:
                    output = await asyncio.to_thread(self.predict_fn, payload)
            except Exception as exc:
                raise InferenceError(
                    f"Inference failed for input '{input_id}'",
                    details={"input_id": input_id},
                ) from exc

            latency_ms = (time.perf_counter() - start) * 1000
            return InferenceResult(input_id=input_id, output=output, latency_ms=latency_ms)

    async def run(self, items: list[tuple[str, Any]]) -> list[InferenceResult]:
        tasks = [self._predict_one(input_id, payload) for input_id, payload in items]
        results = await asyncio.gather(*tasks)
        logger.info("batch_inference_complete", count=len(results))
        return list(results)
