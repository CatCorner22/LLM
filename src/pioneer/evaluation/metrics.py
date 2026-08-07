"""Standard ML/LLM evaluation metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from pioneer.core.registry import Registry

MetricFn = Callable[[list[Any], list[Any]], float]
METRIC_REGISTRY: Registry[MetricFn] = Registry("metric")


def exact_match(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if not predictions:
        return 0.0
    matches = sum(
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references, strict=True)
    )
    return matches / len(predictions)


def f1_score(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    scores: list[float] = []
    for pred, ref in zip(predictions, references, strict=True):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            scores.append(0.0)
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Compute pass@k for code generation benchmarks."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


@METRIC_REGISTRY.register("exact_match")
def _exact_match_metric(predictions: list[Any], references: list[Any]) -> float:
    return exact_match([str(p) for p in predictions], [str(r) for r in references])


@METRIC_REGISTRY.register("f1")
def _f1_metric(predictions: list[Any], references: list[Any]) -> float:
    return f1_score([str(p) for p in predictions], [str(r) for r in references])


class MetricRegistry:
    """Evaluate predictions against references using registered metrics."""

    @staticmethod
    def compute(name: str, predictions: list[Any], references: list[Any]) -> float:
        metric = METRIC_REGISTRY.get(name)
        return metric(predictions, references)

    @staticmethod
    def available() -> list[str]:
        return METRIC_REGISTRY.list()
