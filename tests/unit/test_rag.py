"""Unit tests for RAG pipeline."""

import pytest

from pioneer.models.llm.base import LLMResponse
from pioneer.rag.pipeline import Document, RAGConfig, RAGPipeline, cosine_similarity


class _StubProvider:
    async def complete(self, _request: object) -> LLMResponse:
        return LLMResponse(content="stub answer", model="stub")


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
