"""In-memory RAG pipeline with pluggable retriever and generator."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from pioneer.core.exceptions import RAGError
from pioneer.core.logging import get_logger
from pioneer.models.llm.base import LLMMessage, LLMRequest, LLMRole
from pioneer.models.llm.providers import LLMProvider

logger = get_logger(__name__)


class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] | None = None


class RAGConfig(BaseModel):
    top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)
    system_prompt: str = (
        "Answer the question using only the provided context. "
        "If the context is insufficient, say you don't know."
    )


@dataclass
class RetrievalResult:
    query: str
    documents: list[Document]
    scores: list[float] = field(default_factory=list)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGPipeline:
    """End-to-end retrieval-augmented generation."""

    def __init__(
        self,
        config: RAGConfig,
        provider: LLMProvider,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.embed_fn = embed_fn
        self._documents: list[Document] = []

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)

    def _chunk_text(self, text: str) -> list[str]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += size - overlap
        return chunks

    async def index(self, raw_documents: list[tuple[str, str, dict[str, str]]]) -> int:
        """Index raw documents as (id, content, metadata) tuples."""
        indexed: list[Document] = []
        for doc_id, content, metadata in raw_documents:
            for chunk_index, chunk in enumerate(self._chunk_text(content)):
                indexed.append(
                    Document(
                        id=f"{doc_id}#{chunk_index}",
                        content=chunk,
                        metadata=metadata,
                    )
                )

        if self.embed_fn:
            embeddings = await self._embed_async([doc.content for doc in indexed])
            for doc, embedding in zip(indexed, embeddings, strict=True):
                doc.embedding = embedding

        self._documents.extend(indexed)
        logger.info("rag_indexed", count=len(indexed))
        return len(indexed)

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        if self.embed_fn is None:
            raise RAGError("Embedding function not configured")
        if not callable(self.embed_fn):
            raise RAGError("Invalid embedding function")
        result = self.embed_fn(texts)
        if hasattr(result, "__await__"):
            awaited: list[list[float]] = await result
            return awaited
        if isinstance(result, list):
            return result
        raise RAGError("Invalid embedding function result")

    async def retrieve(self, query: str) -> RetrievalResult:
        if not self._documents:
            return RetrievalResult(query=query, documents=[])

        if self.embed_fn and all(doc.embedding for doc in self._documents):
            query_embedding = (await self._embed_async([query]))[0]
            scored = [
                (doc, cosine_similarity(query_embedding, doc.embedding or []))
                for doc in self._documents
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            top = scored[: self.config.top_k]
            return RetrievalResult(
                query=query,
                documents=[doc for doc, _ in top],
                scores=[score for _, score in top],
            )

        query_lower = query.lower()
        scored = [
            (doc, sum(1 for token in query_lower.split() if token in doc.content.lower()))
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[: self.config.top_k]
        return RetrievalResult(
            query=query,
            documents=[doc for doc, _ in top],
            scores=[float(score) for _, score in top],
        )

    async def generate(self, query: str) -> str:
        retrieval = await self.retrieve(query)
        if not retrieval.documents:
            raise RAGError("No documents indexed for retrieval")

        context = "\n\n".join(f"[{doc.id}] {doc.content}" for doc in retrieval.documents)
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=self.config.system_prompt),
            LLMMessage(
                role=LLMRole.USER,
                content=f"Context:\n{context}\n\nQuestion: {query}",
            ),
        ]
        response = await self.provider.complete(LLMRequest(messages=messages, temperature=0.0))
        logger.info("rag_generated", query=query, sources=len(retrieval.documents))
        return response.content
