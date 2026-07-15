"""Pluggable vector backends for the dual-vector index.

A backend stores (doc_id, summary) pairs and answers "which doc_ids best
match this query". Two implementations ship with tablerag:

- InMemoryBackend: zero-infrastructure default. Embeds summaries itself and
  scores with a configurable similarity metric (cosine / dot / euclidean).
- LangChainVectorStoreBackend: delegates storage, embedding, and search to
  any LangChain-compatible VectorStore (Chroma, FAISS, Qdrant, Pinecone...).
  Duck-typed: langchain does not need to be importable to use this module.

Custom backends only need to satisfy the VectorBackend protocol.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

from tablerag.index.embedder import Embedder

logger = logging.getLogger("tablerag")

SIMILARITY_METHODS = ("cosine", "dot", "euclidean")

DOC_ID_KEY = "doc_id"


class VectorBackend(Protocol):
  """Minimal contract for a summary-vector store."""

  def add(self, ids: list[str], summaries: list[str]) -> None: ...

  def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
    """Return up to top_k (doc_id, score) pairs, higher score = better."""
    ...

  def __len__(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(y * y for y in b))
  if norm_a == 0 or norm_b == 0:
    return 0.0
  return dot / (norm_a * norm_b)


def _dot(a: list[float], b: list[float]) -> float:
  return sum(x * y for x, y in zip(a, b))


def _euclidean_similarity(a: list[float], b: list[float]) -> float:
  """Euclidean distance mapped to (0, 1]: 1 / (1 + distance)."""
  dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
  return 1.0 / (1.0 + dist)


_SCORERS = {
  "cosine": _cosine,
  "dot": _dot,
  "euclidean": _euclidean_similarity,
}


_NO_EMBEDDER_MSG = (
  "No embedder configured. This operation needs embeddings — pass embedder= "
  "(or vectorstore=/vector_backend=) when constructing, e.g. "
  "tablerag.providers.gemini_embedder() or "
  "tablerag.providers.langchain_embedder(OpenAIEmbeddings())."
)


class InMemoryBackend:
  """Embeds summaries via a tablerag Embedder; brute-force scored search.

  The embedder may be omitted at construction; the requirement is enforced
  lazily the first time embedding is actually needed (add/search), so pure
  stages like parse/chunk work without one.
  """

  def __init__(self, embedder: Embedder | None = None, similarity: str = "cosine") -> None:
    if similarity not in SIMILARITY_METHODS:
      raise ValueError(
        f"similarity must be one of {SIMILARITY_METHODS}, got {similarity!r}"
      )
    self.embedder = embedder
    self.similarity = similarity
    self._scorer = _SCORERS[similarity]
    self._ids: list[str] = []
    self._vectors: list[list[float]] = []

  def _require_embedder(self) -> Embedder:
    if self.embedder is None:
      raise ValueError(_NO_EMBEDDER_MSG)
    return self.embedder

  def add(self, ids: list[str], summaries: list[str]) -> None:
    if not ids:
      return
    vectors = self._require_embedder().embed(summaries)
    self._ids.extend(ids)
    self._vectors.extend(vectors)

  def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
    if not self._ids:
      return []
    query_vec = self._require_embedder().embed([query])[0]
    scored = [
      (doc_id, self._scorer(query_vec, vec))
      for doc_id, vec in zip(self._ids, self._vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]

  def __len__(self) -> int:
    return len(self._ids)


class LangChainVectorStoreBackend:
  """Delegates to a LangChain VectorStore (which embeds with its own
  configured embedding function).

  Score caveat: `similarity_search_with_relevance_scores` (normalized 0-1,
  higher = better) is preferred so hybrid lexical blending stays meaningful.
  Stores that only implement raw `similarity_search_with_score` may return
  distances instead; blending quality then depends on the store.
  """

  def __init__(self, vectorstore) -> None:
    self.vectorstore = vectorstore
    self._count = 0

  def add(self, ids: list[str], summaries: list[str]) -> None:
    if not ids:
      return
    metadatas = [{DOC_ID_KEY: doc_id} for doc_id in ids]
    self.vectorstore.add_texts(summaries, metadatas=metadatas)
    self._count += len(ids)

  def _raw_search(self, query: str, top_k: int) -> list[tuple[object, float]]:
    try:
      return self.vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
      )
    except (NotImplementedError, AttributeError, ValueError):
      pass
    try:
      return self.vectorstore.similarity_search_with_score(query, k=top_k)
    except (NotImplementedError, AttributeError):
      docs = self.vectorstore.similarity_search(query, k=top_k)
      return [(doc, 0.0) for doc in docs]

  def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
    results = []
    for doc, score in self._raw_search(query, top_k):
      doc_id = doc.metadata.get(DOC_ID_KEY)
      if doc_id:
        results.append((doc_id, float(score)))
    return results

  def __len__(self) -> int:
    return self._count
