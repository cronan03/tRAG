"""Dual-vector indexing: summary embeddings pointing at raw block payloads."""

from tablerag.index.backends import (
  InMemoryBackend,
  LangChainVectorStoreBackend,
  VectorBackend,
)
from tablerag.index.docstore import DocStore
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import GeminiEmbedder, HashEmbedder

__all__ = [
  "DocStore",
  "DualVectorIndex",
  "GeminiEmbedder",
  "HashEmbedder",
  "InMemoryBackend",
  "LangChainVectorStoreBackend",
  "VectorBackend",
]
