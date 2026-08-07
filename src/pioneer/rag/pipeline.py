"""In-memory RAG pipeline with pluggable retriever and generator.

Retrieval draws on several well-established techniques:

* Reciprocal Rank Fusion (RRF) for combining lexical and dense rankings —
  Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet
  and Individual Rank Learning Methods" (SIGIR 2009).
* Maximal Marginal Relevance (MMR) for diversity-aware reranking —
  Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for
  Reordering Documents and Producing Summaries" (SIGIR 1998).
* Hypothetical Document Embeddings (HyDE) for zero-shot dense retrieval —
  Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  (arXiv:2212.10496).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from pioneer.core.exceptions import RAGError
from pioneer.core.logging import get_logger
from pioneer.models.llm.base import LLMMessage, LLMRequest, LLMRole
from pioneer.models.llm.providers import LLMProvider

logger = get_logger(__name__)

RetrievalStrategy = Literal["dense", "keyword", "hybrid"]


class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] | None = None


class RAGConfig(BaseModel):
    top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)
    retrieval_strategy: RetrievalStrategy = "hybrid"
    rrf_k: int = Field(default=60, ge=1)
    candidate_multiplier: int = Field(default=4, ge=1)
    use_mmr: bool = False
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    use_hyde: bool = False
    hyde_prompt: str = (
        "Write a short, factual passage that would answer the following question. "
        "Do not add commentary."
    )
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


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked ID lists into combined RRF scores.

    Each ranking contributes ``1 / (k + rank)`` (rank is 1-based) to a document's
    score, so documents ranked highly by several retrievers rise to the top
    without any score normalization. See Cormack et al. (SIGIR 2009).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def maximal_marginal_relevance(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    *,
    lambda_mult: float = 0.5,
    top_k: int,
) -> list[int]:
    """Return indices selected by Maximal Marginal Relevance.

    MMR balances query relevance against novelty relative to already-selected
    documents, reducing redundancy in the returned set (Carbonell & Goldstein,
    SIGIR 1998).
    """
    if not doc_embeddings:
        return []

    relevance = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    candidates = list(range(len(doc_embeddings)))
    selected: list[int] = []

    while candidates and len(selected) < top_k:
        best_idx = candidates[0]
        best_score = -math.inf
        for idx in candidates:
            if selected:
                redundancy = max(
                    cosine_similarity(doc_embeddings[idx], doc_embeddings[s]) for s in selected
                )
            else:
                redundancy = 0.0
            score = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


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

    def _embeddings_available(self) -> bool:
        return bool(self.embed_fn) and all(doc.embedding for doc in self._documents)

    async def _hyde_query_embedding(self, query: str) -> list[float]:
        """Embed a hypothetical answer to the query instead of the query itself."""
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=self.config.hyde_prompt),
            LLMMessage(role=LLMRole.USER, content=query),
        ]
        response = await self.provider.complete(LLMRequest(messages=messages, temperature=0.0))
        logger.info("rag_hyde", query=query)
        return (await self._embed_async([response.content]))[0]

    async def _dense_ranking(self, query: str) -> list[tuple[str, float]]:
        if self.config.use_hyde:
            query_embedding = await self._hyde_query_embedding(query)
        else:
            query_embedding = (await self._embed_async([query]))[0]
        scored = [
            (doc.id, cosine_similarity(query_embedding, doc.embedding or []))
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _keyword_ranking(self, query: str) -> list[tuple[str, float]]:
        query_tokens = query.lower().split()
        scored = [
            (
                doc.id,
                float(sum(1 for token in query_tokens if token in doc.content.lower())),
            )
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _apply_mmr(self, query_embedding: list[float], documents: list[Document]) -> list[Document]:
        embeddings = [doc.embedding or [] for doc in documents]
        order = maximal_marginal_relevance(
            query_embedding,
            embeddings,
            lambda_mult=self.config.mmr_lambda,
            top_k=self.config.top_k,
        )
        return [documents[i] for i in order]

    async def retrieve(self, query: str) -> RetrievalResult:
        if not self._documents:
            return RetrievalResult(query=query, documents=[])

        by_id = {doc.id: doc for doc in self._documents}
        dense_available = self._embeddings_available()
        strategy = self.config.retrieval_strategy

        use_dense = strategy in ("dense", "hybrid") and dense_available
        use_keyword = strategy in ("keyword", "hybrid") or not dense_available

        rankings: list[list[str]] = []
        score_lookup: dict[str, float] = {}

        if use_dense:
            dense = await self._dense_ranking(query)
            rankings.append([doc_id for doc_id, _ in dense])
            score_lookup = dict(dense)
        if use_keyword:
            keyword = self._keyword_ranking(query)
            rankings.append([doc_id for doc_id, _ in keyword])
            if not score_lookup:
                score_lookup = dict(keyword)

        if not rankings:
            return RetrievalResult(query=query, documents=[])

        if len(rankings) > 1:
            fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k)
            ordered_ids = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)
            score_lookup = fused
        else:
            ordered_ids = rankings[0]

        pool_size = self.config.top_k * self.config.candidate_multiplier
        candidate_ids = ordered_ids[:pool_size]
        candidates = [by_id[doc_id] for doc_id in candidate_ids]

        if self.config.use_mmr and dense_available:
            if self.config.use_hyde:
                query_embedding = await self._hyde_query_embedding(query)
            else:
                query_embedding = (await self._embed_async([query]))[0]
            top_documents = self._apply_mmr(query_embedding, candidates)
        else:
            top_documents = candidates[: self.config.top_k]

        return RetrievalResult(
            query=query,
            documents=top_documents,
            scores=[score_lookup.get(doc.id, 0.0) for doc in top_documents],
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
