"""Dual-vector index: embed summaries for search, resolve doc_id -> raw block.

The vector side holds only searchable summaries; the docstore holds the
pristine payloads. Retrieval returns raw blocks, never summaries.

Vector storage/search is delegated to a pluggable VectorBackend
(in-memory by default, or any external store). On top of the backend's
semantic score, an IDF-weighted exact-token overlap is blended in, because
table queries lean on identifiers (column names, SKUs, dates) that
embeddings alone blur. Set lexical_weight=0 to disable.
"""

from __future__ import annotations

import logging

from tablerag.index.backends import InMemoryBackend, VectorBackend
from tablerag.index.docstore import DocStore
from tablerag.index.embedder import Embedder
from tablerag.index.lexical import LexicalScorer
from tablerag.models import Block, RetrievedBlock
from tablerag.summarize import summarize_block

logger = logging.getLogger("tablerag")

# How many semantic candidates to pull from the backend before lexical
# re-ranking. Generous so a lexically strong block semantically ranked low
# can still be rescued.
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 50


class DualVectorIndex:
  def __init__(
    self,
    embedder: Embedder | None = None,
    lexical_weight: float = 0.5,
    backend: VectorBackend | None = None,
    similarity: str = "cosine",
  ) -> None:
    """
    Args:
      embedder: tablerag Embedder; required when no backend is given.
      lexical_weight: 0..1 blend of lexical overlap vs semantic score.
      backend: custom VectorBackend (e.g. LangChainVectorStoreBackend).
        Defaults to InMemoryBackend(embedder, similarity).
      similarity: "cosine" | "dot" | "euclidean" (in-memory backend only).
    """
    if backend is None:
      if embedder is None:
        raise ValueError("Provide an embedder or a backend.")
      backend = InMemoryBackend(embedder, similarity=similarity)
    self.backend = backend
    self.embedder = embedder
    self.lexical_weight = lexical_weight
    self.docstore = DocStore()
    self.lexical = LexicalScorer()
    self._lexical_index: dict[str, int] = {}  # doc_id -> LexicalScorer index
    self._summaries: dict[str, str] = {}

  def add_blocks(self, blocks: list[Block]) -> None:
    if not blocks:
      return
    summaries = [summarize_block(b) for b in blocks]
    ids = [b.doc_id for b in blocks]

    for block, summary in zip(blocks, summaries):
      self.docstore.put(block)
      self._summaries[block.doc_id] = summary
      self._lexical_index[block.doc_id] = len(self._lexical_index)

    self.backend.add(ids, summaries)
    self.lexical.add_documents(summaries)
    logger.info("Indexed %d blocks (total %d)", len(blocks), len(self.backend))

  def search(self, query: str, top_k: int = 3) -> list[RetrievedBlock]:
    if len(self.backend) == 0:
      return []

    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    candidates = self.backend.search(query, candidate_k)

    w = self.lexical_weight
    scored: list[tuple[str, float]] = []
    for doc_id, semantic in candidates:
      lexical = (
        self.lexical.score(query, self._lexical_index[doc_id])
        if w > 0 and doc_id in self._lexical_index
        else 0.0
      )
      scored.append((doc_id, (1 - w) * semantic + w * lexical))
    scored.sort(key=lambda item: item[1], reverse=True)

    return [
      RetrievedBlock(block=self.docstore.get(doc_id), score=score)
      for doc_id, score in scored[:top_k]
      if doc_id in self.docstore
    ]

  def all_scores(self, query: str) -> list[RetrievedBlock]:
    return self.search(query, top_k=len(self.backend))

  def summary_for(self, doc_id: str) -> str:
    return self._summaries.get(doc_id, "")

  def __len__(self) -> int:
    return len(self.backend)
