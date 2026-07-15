"""One-line provider constructors.

tablerag is model-agnostic: the pipeline needs a `Generator` and an `Embedder`
(or a vector backend/store that embeds itself). This module offers convenient
constructors for common setups, but you can always pass your own objects.

Gemini (native, zero extra deps):
    from tablerag.providers import gemini_generator, gemini_embedder
    pipe = TableRAGPipeline(generator=gemini_generator(), embedder=gemini_embedder())

Any LangChain model (OpenAI, Anthropic, Gemini, Cohere, local, ...):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from tablerag.providers import langchain_generator, langchain_embedder
    pipe = TableRAGPipeline(
        generator=langchain_generator(ChatOpenAI(model="gpt-4o")),
        embedder=langchain_embedder(OpenAIEmbeddings(model="text-embedding-3-small")),
    )

Anything else:
    from tablerag.generate import CallableGenerator
    from tablerag.index.embedder import CallableEmbedder
    pipe = TableRAGPipeline(
        generator=CallableGenerator(my_llm_call),
        embedder=CallableEmbedder(my_embed_call),
    )
"""

from __future__ import annotations

from tablerag.generate import (
  CallableGenerator,
  GeminiGenerator,
  Generator,
  LangChainGenerator,
)
from tablerag.index.embedder import (
  CallableEmbedder,
  Embedder,
  GeminiEmbedder,
  LangChainEmbedder,
)

__all__ = [
  "gemini_generator",
  "gemini_embedder",
  "langchain_generator",
  "langchain_embedder",
  "CallableGenerator",
  "CallableEmbedder",
  "Generator",
  "Embedder",
]


def gemini_generator(model: str | None = None, api_key: str | None = None) -> Generator:
  """Native Gemini answer generator (reads GEMINI_API_KEY / GEMINI_MODEL)."""
  return GeminiGenerator(model=model, api_key=api_key)


def gemini_embedder(model: str | None = None, api_key: str | None = None) -> Embedder:
  """Native Gemini embedder (reads GEMINI_API_KEY / GEMINI_EMBED_MODEL)."""
  return GeminiEmbedder(model=model, api_key=api_key)


def langchain_generator(chat_model, model: str | None = None) -> Generator:
  """Adapt any LangChain chat model (ChatOpenAI, ChatAnthropic, ...)."""
  return LangChainGenerator(chat_model, model=model)


def langchain_embedder(embeddings) -> Embedder:
  """Adapt any LangChain Embeddings (OpenAIEmbeddings, CohereEmbeddings, ...)."""
  return LangChainEmbedder(embeddings)
