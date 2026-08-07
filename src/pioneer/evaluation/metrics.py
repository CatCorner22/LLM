"""Standard ML/LLM evaluation metrics.

The token-overlap metrics follow the normalization and F1 definitions from
Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
(arXiv:1606.05250), and ``pass_at_k`` uses the unbiased estimator from Chen et
al., "Evaluating Large Language Models Trained on Code" (arXiv:2107.03374).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable
from typing import Any

from pioneer.core.registry import Registry

MetricFn = Callable[[list[Any], list[Any]], float]
METRIC_REGISTRY: Registry[MetricFn] = Registry("metric")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalize an answer using the canonical SQuAD procedure.

    Lowercases, removes punctuation and the articles ``a``/``an``/``the``, and
    collapses whitespace so that surface differences do not penalize otherwise
    correct answers (Rajpurkar et al., arXiv:1606.05250).
    """
    lowered = text.lower()
    without_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = _ARTICLES.sub(" ", without_punct)
    return _WHITESPACE.sub(" ", without_articles).strip()


def exact_match(predictions: list[str], references: list[str], *, normalize: bool = False) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if not predictions:
        return 0.0

    def prepare(value: str) -> str:
        return normalize_answer(value) if normalize else value.strip().lower()

    matches = sum(
        prepare(pred) == prepare(ref) for pred, ref in zip(predictions, references, strict=True)
    )
    return matches / len(predictions)


def _tokenize(value: str, *, normalize: bool) -> list[str]:
    return normalize_answer(value).split() if normalize else value.lower().split()


def f1_score(predictions: list[str], references: list[str], *, normalize: bool = False) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    scores: list[float] = []
    for pred, ref in zip(predictions, references, strict=True):
        pred_tokens = _tokenize(pred, normalize=normalize)
        ref_tokens = _tokenize(ref, normalize=normalize)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0 or not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def squad_exact_match(predictions: list[str], references: list[str]) -> float:
    """SQuAD-normalized exact match (arXiv:1606.05250)."""
    return exact_match(predictions, references, normalize=True)


def squad_f1(predictions: list[str], references: list[str]) -> float:
    """SQuAD-normalized token-level F1 (arXiv:1606.05250)."""
    return f1_score(predictions, references, normalize=True)


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


@METRIC_REGISTRY.register("squad_em")
def _squad_em_metric(predictions: list[Any], references: list[Any]) -> float:
    return squad_exact_match([str(p) for p in predictions], [str(r) for r in references])


@METRIC_REGISTRY.register("squad_f1")
def _squad_f1_metric(predictions: list[Any], references: list[Any]) -> float:
    return squad_f1([str(p) for p in predictions], [str(r) for r in references])


class MetricRegistry:
    """Evaluate predictions against references using registered metrics."""

    @staticmethod
    def compute(name: str, predictions: list[Any], references: list[Any]) -> float:
        metric = METRIC_REGISTRY.get(name)
        return metric(predictions, references)

    @staticmethod
    def available() -> list[str]:
        return METRIC_REGISTRY.list()
