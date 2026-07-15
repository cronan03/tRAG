"""Provider-agnostic generation & embedding: adapters and no-default guards."""

from __future__ import annotations

import pytest

from tablerag import TableRAGPipeline
from tablerag.generate import (
  CallableGenerator,
  Generator,
  LangChainGenerator,
)
from tablerag.index.embedder import (
  CallableEmbedder,
  Embedder,
  HashEmbedder,
  LangChainEmbedder,
)

DOC = """=== SALES ===
Weekly revenue by region.

| region | net_rev_usd |
| --- | --- |
| NA-EAST | 640220.10 |
| NA-WEST | 325044.80 |
"""


# --------------------------------------------------------------- adapters
def test_callable_generator_is_a_generator():
  gen = CallableGenerator(lambda p: f"echo:{len(p)}", model="my-model")
  assert isinstance(gen, Generator)
  assert gen.model == "my-model"
  assert gen.generate("hello") == "echo:5"


def test_callable_embedder_is_an_embedder():
  emb = CallableEmbedder(lambda texts: [[float(len(t))] for t in texts])
  assert isinstance(emb, Embedder)
  assert emb.embed(["ab", "abc"]) == [[2.0], [3.0]]
  assert emb.embed([]) == []


class _FakeMessage:
  def __init__(self, content):
    self.content = content


class _FakeChatModel:
  """Mimics a LangChain BaseChatModel (only .invoke is used)."""

  model = "fake-chat"

  def __init__(self, reply="canned answer"):
    self._reply = reply
    self.seen = None

  def invoke(self, prompt):
    self.seen = prompt
    return _FakeMessage(self._reply)


class _FakeEmbeddings:
  """Mimics a LangChain Embeddings object."""

  def embed_documents(self, texts):
    return [[float(len(t)), 1.0] for t in texts]


def test_langchain_generator_extracts_content():
  chat = _FakeChatModel(reply="the answer")
  gen = LangChainGenerator(chat)
  assert isinstance(gen, Generator)
  assert gen.model == "fake-chat"
  assert gen.generate("prompt text") == "the answer"
  assert chat.seen == "prompt text"


def test_langchain_generator_handles_block_list_content():
  chat = _FakeChatModel(reply=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
  gen = LangChainGenerator(chat)
  assert gen.generate("x") == "ab"


def test_langchain_embedder_delegates():
  emb = LangChainEmbedder(_FakeEmbeddings())
  assert isinstance(emb, Embedder)
  assert emb.embed(["ab"]) == [[2.0, 1.0]]


# --------------------------------------------------------------- no defaults
def test_pipeline_defers_embedder_error_until_embedding():
  # No embedder + no vectorstore: construction and the pure stages work...
  pipe = TableRAGPipeline(enable_compute=False)
  blocks = pipe.parse(DOC)
  chunks = pipe.chunk(blocks)
  assert chunks
  # ...but embedding (index_blocks) raises a clear, actionable error.
  with pytest.raises(ValueError, match="No embedder configured"):
    pipe.index_blocks(chunks)


def test_pipeline_generator_required_only_for_generation():
  # Retrieval-only is fine without a generator...
  pipe = TableRAGPipeline(embedder=HashEmbedder(), enable_compute=False)
  pipe.ingest(DOC)
  hits = pipe.index.search("NA-WEST revenue", top_k=2)
  assert hits
  # ...but touching generation raises a clear, actionable error.
  with pytest.raises(ValueError, match="No generator configured"):
    _ = pipe.generator


# ------------------------------------------------------- end-to-end agnostic
def test_end_to_end_with_arbitrary_provider():
  """A fully non-Gemini pipeline: callable generator + hash embedder."""
  captured = {}

  def fake_llm(prompt: str) -> str:
    captured["prompt"] = prompt
    return "NA-WEST net revenue is 325044.80 USD."

  pipe = TableRAGPipeline(
    generator=CallableGenerator(fake_llm, model="unit-test-llm"),
    embedder=HashEmbedder(),
    enable_compute=False,
  )
  pipe.ingest(DOC)
  result = pipe.query("What is NA-WEST net revenue?", top_k=2)

  assert result.answer == "NA-WEST net revenue is 325044.80 USD."
  assert "325044.80" in captured["prompt"]  # retrieved table reached the LLM
  assert pipe.model == "unit-test-llm"
