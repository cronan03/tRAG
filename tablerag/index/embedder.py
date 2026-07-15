"""Embedding backends, provider-agnostic.

Anything satisfying the `Embedder` protocol (`embed(texts) -> list[list[float]]`)
works:

- GeminiEmbedder     native google-genai (zero extra deps)
- LangChainEmbedder  wraps any LangChain Embeddings (OpenAI, Cohere, HF, ...)
- CallableEmbedder   wraps any `fn(texts) -> list[list[float]]`
- HashEmbedder       deterministic offline embedder for tests / quota-free runs

See tablerag.providers for one-line constructors.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger("tablerag")

DEFAULT_EMBED_MODEL = "gemini-embedding-001"


@runtime_checkable
class Embedder(Protocol):
  def embed(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
  """Embeds text via the google-genai SDK. Batches all inputs in one call."""

  def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    self.model = model or os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    self._api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not self._api_key:
      raise EnvironmentError("GEMINI_API_KEY is not set.")
    self._client = None

  @property
  def client(self):
    if self._client is None:
      from google import genai

      self._client = genai.Client(api_key=self._api_key)
    return self._client

  def embed(self, texts: list[str]) -> list[list[float]]:
    if not texts:
      return []
    logger.info("Embedding %d texts with %s", len(texts), self.model)
    response = self.client.models.embed_content(model=self.model, contents=texts)
    return [list(e.values) for e in response.embeddings]


class LangChainEmbedder:
  """Wraps any LangChain Embeddings object as an Embedder.

  Duck-typed: only `.embed_documents()` is required, so langchain need not be
  importable here.
  """

  def __init__(self, embeddings) -> None:
    self.embeddings = embeddings

  def embed(self, texts: list[str]) -> list[list[float]]:
    if not texts:
      return []
    return self.embeddings.embed_documents(texts)


class CallableEmbedder:
  """Wraps any `fn(texts: list[str]) -> list[list[float]]` as an Embedder."""

  def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
    self._fn = fn

  def embed(self, texts: list[str]) -> list[list[float]]:
    if not texts:
      return []
    return self._fn(texts)


class HashEmbedder:
  """Deterministic, offline bag-of-words hashing embedder.

  Not semantically smart, but strictly better than nothing for unit tests and
  quota-free local runs: shared tokens produce overlapping dimensions.
  """

  def __init__(self, dim: int = 512) -> None:
    self.dim = dim

  def _tokens(self, text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-\.]+", text.lower())

  def embed(self, texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
      vec = [0.0] * self.dim
      for token in self._tokens(text):
        digest = hashlib.md5(token.encode()).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        vec[idx] += 1.0
      norm = math.sqrt(sum(v * v for v in vec)) or 1.0
      vectors.append([v / norm for v in vec])
    return vectors
