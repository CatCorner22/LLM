"""Unit tests for RAG pipeline."""

import pytest

from pioneer.models.llm.base import LLMResponse
from pioneer.rag.pipeline import (
    Document,
    RAGConfig,
    RAGPipeline,
    cosine_similarity,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)


class _StubProvider:
    async def complete(self, _request: object) -> LLMResponse:
        return LLMResponse(content="stub answer", model="stub")


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, _request: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="python is a language for machine learning", model="stub")


@pytest.mark.unit
def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_retrieve_keyword() -> None:
    pipeline = RAGPipeline(config=RAGConfig(top_k=2), provider=_StubProvider())  # type: ignore[arg-type]
    pipeline.add_documents(
        [
            Document(id="1", content="Python is a programming language"),
            Document(id="2", content="JavaScript runs in browsers"),
            Document(id="3", content="Python supports machine learning"),
        ]
    )
    result = await pipeline.retrieve("Python machine learning")
    assert len(result.documents) == 2
    assert any("Python" in doc.content for doc in result.documents)


@pytest.mark.unit
def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # "a" and "b" appear near the top of both lists, so they outrank "c"/"d".
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["d"]
    ordered = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)
    assert set(ordered[:2]) == {"a", "b"}


@pytest.mark.unit
def test_mmr_prefers_diversity_when_lambda_low() -> None:
    query = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]
    # Pure relevance keeps the two near-duplicate top docs.
    assert maximal_marginal_relevance(query, embeddings, lambda_mult=1.0, top_k=2) == [0, 1]
    # Diversity-weighted selection swaps in the orthogonal document.
    assert maximal_marginal_relevance(query, embeddings, lambda_mult=0.3, top_k=2) == [0, 2]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_hybrid_with_embeddings() -> None:
    config = RAGConfig(top_k=2, retrieval_strategy="hybrid")
    pipeline = RAGPipeline(
        config=config,
        provider=_StubProvider(),  # type: ignore[arg-type]
        embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    pipeline.add_documents(
        [
            Document(id="1", content="python programming", embedding=[1.0, 0.0, 0.0]),
            Document(id="2", content="javascript browser", embedding=[0.0, 1.0, 0.0]),
            Document(id="3", content="python machine learning", embedding=[1.0, 1.0, 0.0]),
        ]
    )
    result = await pipeline.retrieve("python")
    assert len(result.documents) == 2
    assert result.documents[0].id == "1"
    assert "2" not in [doc.id for doc in result.documents]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_hyde_invokes_provider() -> None:
    provider = _RecordingProvider()
    config = RAGConfig(top_k=1, retrieval_strategy="dense", use_hyde=True)
    pipeline = RAGPipeline(
        config=config,
        provider=provider,  # type: ignore[arg-type]
        embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    pipeline.add_documents(
        [
            Document(id="1", content="python", embedding=[1.0, 0.0]),
            Document(id="2", content="rust", embedding=[0.0, 1.0]),
        ]
    )
    result = await pipeline.retrieve("which language is best for ML?")
    assert provider.calls == 1  # HyDE generated a hypothetical document
    assert result.documents[0].id == "1"
