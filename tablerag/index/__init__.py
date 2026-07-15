"""Dual-vector indexing: summary embeddings pointing at raw block payloads."""

from tablerag.index.backends import (
  InMemoryBackend,
  LangChainVectorStoreBackend,
  VectorBackend,
)
from tablerag.index.docstore import DocStore
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import (
  CallableEmbedder,
  Embedder,
  GeminiEmbedder,
  HashEmbedder,
  LangChainEmbedder,
)

__all__ = [
  "DocStore",
  "DualVectorIndex",
  "Embedder",
  "GeminiEmbedder",
  "HashEmbedder",
  "LangChainEmbedder",
  "CallableEmbedder",
  "InMemoryBackend",
  "LangChainVectorStoreBackend",
  "VectorBackend",
]
