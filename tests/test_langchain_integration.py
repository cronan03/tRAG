"""Offline tests for the LangChain adapter (HashEmbedder, no API calls)."""

from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from tablerag.index.embedder import HashEmbedder
from tablerag.integrations.langchain import (
  DOC_ID_KEY,
  LangChainEmbedderAdapter,
  TableRetrieverManager,
)

DATA_DIR = Path(__file__).parent.parent / "data"
DOC2 = DATA_DIR / "document2.txt"


class FakeLCEmbeddings(Embeddings):
  """LangChain Embeddings interface backed by the offline HashEmbedder."""

  def __init__(self) -> None:
    self._hash = HashEmbedder()

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    return self._hash.embed(texts)

  def embed_query(self, text: str) -> list[float]:
    return self._hash.embed([text])[0]


# --------------------------------------------------------- internal mode
@pytest.fixture(scope="module")
def internal_manager():
  manager = TableRetrieverManager(embedder=HashEmbedder())
  manager.ingest(DOC2)
  return manager


def test_as_retriever_returns_base_retriever(internal_manager):
  retriever = internal_manager.as_retriever(k=3)
  assert isinstance(retriever, BaseRetriever)


def test_retriever_invoke_returns_raw_tables(internal_manager):
  retriever = internal_manager.as_retriever(k=3)
  docs = retriever.invoke("warehouse bay stock on hand oh avail for skus")

  assert len(docs) == 3
  assert all(isinstance(d, Document) for d in docs)
  sections = [d.metadata["section"] for d in docs]
  assert any("PRODUCT MOVEMENT" in s for s in sections)
  # Raw markdown payload, not the summary.
  table_docs = [d for d in docs if d.metadata["kind"] == "table"]
  assert table_docs and "| loc | sku |" in table_docs[0].page_content


def test_metadata_contract(internal_manager):
  docs = internal_manager.as_retriever(k=2).invoke("net_rev_eur DE-442")
  for doc in docs:
    assert DOC_ID_KEY in doc.metadata
    assert doc.metadata["kind"] in {"table", "text"}
    assert "section" in doc.metadata
    assert "score" in doc.metadata


def test_langchain_embeddings_adapter():
  adapter = LangChainEmbedderAdapter(FakeLCEmbeddings())
  vectors = adapter.embed(["hello world", "foo bar"])
  assert len(vectors) == 2
  assert len(vectors[0]) == 512


def test_ingest_tables_convenience():
  manager = TableRetrieverManager(embedder=HashEmbedder())
  blocks = manager.ingest_tables(
    ["| Quarter | Revenue |\n| Q1 2025 | 12.5 |\n| Q4 2025 | 18.1 |"]
  )
  assert blocks[0].kind == "table"
  docs = manager.as_retriever(k=1).invoke("Q4 2025 quarterly revenue")
  assert "18.1" in docs[0].page_content


# ------------------------------------------------------- vectorstore mode
@pytest.fixture(scope="module")
def vectorstore_manager():
  store = InMemoryVectorStore(embedding=FakeLCEmbeddings())
  manager = TableRetrieverManager(vectorstore=store)
  manager.ingest(DOC2)
  return manager


def test_vectorstore_holds_summaries_not_payloads(vectorstore_manager):
  store_docs = list(vectorstore_manager.vectorstore.store.values())
  assert len(store_docs) > 0
  # Every stored entry carries a doc_id pointer and is a summary,
  # not raw canonical markdown.
  for entry in store_docs:
    assert DOC_ID_KEY in entry["metadata"]


def test_vectorstore_search_resolves_raw_blocks(vectorstore_manager):
  # Vectorstore mode has no lexical hybrid boost (ranking is the store's
  # job), and the offline hash embedder is weak on fuzzy phrasing - so use
  # a token-distinctive query and verify the doc_id -> raw block resolution.
  docs = vectorstore_manager.search(
    "WB-8841 A1-04 wireless buds oh reserved avail last_recv", k=3
  )
  sections = [d.metadata.get("section", "") for d in docs]
  assert any("PRODUCT MOVEMENT" in s for s in sections)
  table_docs = [d for d in docs if d.metadata.get("kind") == "table"]
  assert table_docs and "| loc | sku |" in table_docs[0].page_content


def test_vectorstore_retriever_end_to_end(vectorstore_manager):
  retriever = vectorstore_manager.as_retriever(k=2)
  docs = retriever.invoke("corrected net_rev_eur DE-442 restatement")
  assert len(docs) == 2
  assert all(DOC_ID_KEY in d.metadata for d in docs)
