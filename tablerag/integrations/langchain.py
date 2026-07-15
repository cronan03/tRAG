"""LangChain adapter: tablerag's table-aware ingest + retrieval as a native
LangChain retriever.

This replaces the manual MultiVectorRetriever plumbing (UUID loops, summary
chains, InMemoryStore syncing) with:

    from tablerag.integrations.langchain import TableRetrieverManager

    manager = TableRetrieverManager()          # or vectorstore=my_chroma
    manager.ingest("data/document2.txt")
    retriever = manager.as_retriever(k=3)      # native BaseRetriever

    # drop into any standard LangChain chain
    chain = {"context": retriever, "question": ...} | prompt | llm | parser

Two backend modes:
- Internal (default): tablerag's in-memory DualVectorIndex (hybrid
  semantic + lexical scoring). Zero extra infrastructure.
- User vectorstore: pass any LangChain VectorStore (Chroma, FAISS, ...).
  Summaries are written there; raw blocks stay in tablerag's docstore and
  are resolved back via doc_id metadata (the multi-vector pattern).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
  from langchain_core.callbacks import CallbackManagerForRetrieverRun
  from langchain_core.documents import Document
  from langchain_core.embeddings import Embeddings
  from langchain_core.retrievers import BaseRetriever
  from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover
  raise ImportError(
    "tablerag.integrations.langchain requires langchain-core. "
    "Install it with: pip install langchain-core"
  ) from exc

from tablerag.chunk import split_table_block, split_text_block
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import Embedder, LangChainEmbedder
from tablerag.models import Block, TableBlock, TextBlock
from tablerag.parse import parse_document
from tablerag.pipeline import render_block
from tablerag.summarize import Summarizer, summarize_block

logger = logging.getLogger("tablerag")

DOC_ID_KEY = "doc_id"

# Backwards-compatible alias; the canonical adapter now lives in
# tablerag.index.embedder so the core (non-LangChain) path can use it too.
LangChainEmbedderAdapter = LangChainEmbedder


def _block_to_document(block: Block, score: float | None = None) -> Document:
  metadata: dict[str, Any] = {
    DOC_ID_KEY: block.doc_id,
    "kind": block.kind,
    "section": block.section,
    "source": block.source,
  }
  if score is not None:
    metadata["score"] = score
  return Document(page_content=render_block(block), metadata=metadata)


class TableRAGRetriever(BaseRetriever):
  """Native LangChain retriever backed by a TableRetrieverManager."""

  manager: Any
  k: int = 3

  model_config = {"arbitrary_types_allowed": True}

  def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
  ) -> list[Document]:
    return self.manager.search(query, k=self.k)


class TableRetrieverManager:
  def __init__(
    self,
    vectorstore: VectorStore | None = None,
    embedder: Embedder | Embeddings | None = None,
    max_table_rows: int = 50,
    max_table_words: int | None = None,
    max_text_words: int = 300,
    text_overlap_words: int = 0,
    similarity: str = "cosine",
    lexical_weight: float = 0.5,
    summarizer: Summarizer | None = None,
  ) -> None:
    """
    Args:
      vectorstore: optional LangChain VectorStore for summary embeddings.
        When omitted, tablerag's internal in-memory index is used.
      embedder: tablerag Embedder or LangChain Embeddings (internal mode
        only; a vectorstore embeds with its own configured embeddings).
      max_table_rows: row budget per table slice before header-aware split.
      max_table_words: optional word budget per table slice.
      max_text_words: word budget per prose chunk (default 300).
      text_overlap_words: words repeated between consecutive prose chunks.
      similarity: "cosine" | "dot" | "euclidean" (internal mode only).
      lexical_weight: 0..1 lexical/semantic blend (internal mode only;
        vectorstore mode ranks with the store's own similarity).
      summarizer: callable `(Block) -> str` for searchable summaries.
        Defaults to summarize_block. Same semantics as TableRAGPipeline.
    """
    self.max_table_rows = max_table_rows
    self.max_table_words = max_table_words
    self.max_text_words = max_text_words
    self.text_overlap_words = text_overlap_words
    self.vectorstore = vectorstore
    self.summarizer = summarizer or summarize_block

    if vectorstore is not None:
      # Raw blocks live in our docstore; the user's vectorstore only ever
      # sees summaries + doc_id pointers.
      from tablerag.index.docstore import DocStore

      self.index = None
      self.docstore = DocStore()
    else:
      # embedder is optional at construction; DualVectorIndex enforces it
      # lazily when embedding is first needed (ingest/search).
      resolved: Embedder | None
      if embedder is None:
        resolved = None
      elif isinstance(embedder, Embeddings):
        resolved = LangChainEmbedder(embedder)
      else:
        resolved = embedder
      self.index = DualVectorIndex(
        resolved,
        lexical_weight=lexical_weight,
        similarity=similarity,
        summarizer=self.summarizer,
      )
      self.docstore = self.index.docstore

  # ------------------------------------------------------------- ingest
  def ingest(self, path_or_text: str | Path) -> list[Block]:
    """Parse, chunk, summarize, and index a document (file path or text)."""
    path = Path(path_or_text)
    if isinstance(path_or_text, Path) or (
      len(str(path_or_text)) < 4096 and path.exists()
    ):
      text = path.read_text(encoding="utf-8")
      source = path.name
    else:
      text = str(path_or_text)
      source = "(inline)"

    blocks = parse_document(text, source=source)
    chunked: list[Block] = []
    for block in blocks:
      if isinstance(block, TableBlock):
        chunked.extend(
          split_table_block(
            block,
            max_rows=self.max_table_rows,
            max_words=self.max_table_words,
          )
        )
      elif isinstance(block, TextBlock):
        chunked.extend(
          split_text_block(
            block,
            max_words=self.max_text_words,
            overlap_words=self.text_overlap_words,
          )
        )
      else:
        chunked.append(block)

    if self.index is not None:
      self.index.add_blocks(chunked)
    else:
      summary_docs = []
      for block in chunked:
        self.docstore.put(block)
        summary_docs.append(
          Document(
            page_content=self.summarizer(block),
            metadata={DOC_ID_KEY: block.doc_id, "kind": block.kind},
          )
        )
      self.vectorstore.add_documents(summary_docs)

    logger.info(
      "LangChain adapter: ingested %d blocks from %s (%s backend)",
      len(chunked),
      source,
      "vectorstore" if self.index is None else "internal",
    )
    return chunked

  def ingest_tables(self, raw_tables: list[str]) -> list[Block]:
    """Convenience: ingest a list of raw table strings (markdown/TSV/CSV)."""
    ingested: list[Block] = []
    for raw in raw_tables:
      ingested.extend(self.ingest(raw))
    return ingested

  # ------------------------------------------------------------- search
  def search(self, query: str, k: int = 3) -> list[Document]:
    """Retrieve raw blocks (as LangChain Documents), never summaries."""
    if self.index is not None:
      retrieved = self.index.search(query, top_k=k)
      return [_block_to_document(r.block, r.score) for r in retrieved]

    try:
      hits = self.vectorstore.similarity_search_with_score(query, k=k)
    except (NotImplementedError, AttributeError):
      hits = [(doc, None) for doc in self.vectorstore.similarity_search(query, k=k)]

    documents: list[Document] = []
    for summary_doc, score in hits:
      doc_id = summary_doc.metadata.get(DOC_ID_KEY)
      if doc_id and doc_id in self.docstore:
        documents.append(_block_to_document(self.docstore.get(doc_id), score))
      else:
        documents.append(summary_doc)  # foreign doc in a shared store
    return documents

  def as_retriever(self, k: int = 3) -> TableRAGRetriever:
    """Return a native LangChain BaseRetriever for use in any chain."""
    return TableRAGRetriever(manager=self, k=k)
