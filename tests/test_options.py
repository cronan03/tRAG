"""Tests for user-configurable SDK options (all offline)."""

from pathlib import Path

import pytest

from tablerag.chunk import split_table_block, split_text_block
from tablerag.index.backends import InMemoryBackend, LangChainVectorStoreBackend
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import HashEmbedder
from tablerag.models import TableBlock, TextBlock
from tablerag.parse.tables import to_canonical_markdown
from tablerag.pipeline import TableRAGPipeline

DATA_DIR = Path(__file__).parent.parent / "data"
DOC2 = DATA_DIR / "document2.txt"


# ------------------------------------------------------ text splitting
def make_text(num_words: int) -> TextBlock:
  sentences = []
  word = 0
  while word < num_words:
    length = min(10, num_words - word)
    sentences.append(" ".join(f"w{word + i}" for i in range(length)) + ".")
    word += length
  return TextBlock(content=" ".join(sentences), section="LONG SECTION")


def test_short_text_not_split():
  block = make_text(100)
  assert split_text_block(block, max_words=300) == [block]


def test_long_text_split_by_word_budget():
  block = make_text(900)
  slices = split_text_block(block, max_words=300)
  assert len(slices) == 3
  for s in slices:
    assert len(s.content.split()) <= 300
    assert "LONG SECTION (part" in s.section
  # No words lost.
  total = sum(len(s.content.split()) for s in slices)
  assert total == 900


def test_text_overlap_words():
  block = make_text(600)
  slices = split_text_block(block, max_words=300, overlap_words=50)
  first_words = slices[0].content.split()
  second_words = slices[1].content.split()
  assert second_words[:50] == first_words[-50:]


def test_text_overlap_must_be_smaller_than_budget():
  with pytest.raises(ValueError):
    split_text_block(make_text(600), max_words=100, overlap_words=100)


# --------------------------------------------------- table word budget
def make_table(num_rows: int) -> TableBlock:
  headers = ["id", "value"]
  rows = [[str(i), str(i * 10)] for i in range(num_rows)]
  return TableBlock(
    markdown=to_canonical_markdown(headers, rows),
    headers=headers,
    rows=rows,
    section="BIG TABLE",
  )


def test_table_max_words_derives_row_cap():
  table = make_table(100)  # 2 words per row
  slices = split_table_block(table, max_rows=50, max_words=20)
  # ~(20 - 2 header words) / 2 words per row = 9 rows per slice
  assert all(s.num_rows <= 9 for s in slices)
  assert sum(s.num_rows for s in slices) == 100


def test_table_max_words_ignored_when_loose():
  table = make_table(10)
  assert len(split_table_block(table, max_rows=50, max_words=10000)) == 1


# ------------------------------------------------- similarity methods
@pytest.mark.parametrize("similarity", ["cosine", "dot", "euclidean"])
def test_similarity_methods_rank_matching_doc_first(similarity):
  backend = InMemoryBackend(HashEmbedder(), similarity=similarity)
  backend.add(
    ["a", "b"],
    ["warehouse inventory stock oh avail sku", "marketing campaign revenue email"],
  )
  results = backend.search("warehouse stock sku", top_k=2)
  assert results[0][0] == "a"


def test_invalid_similarity_rejected():
  with pytest.raises(ValueError):
    InMemoryBackend(HashEmbedder(), similarity="manhattan")


def test_index_requires_embedder_or_backend():
  with pytest.raises(ValueError):
    DualVectorIndex()


def test_lexical_weight_zero_equals_raw_backend_score():
  block = TextBlock(content="alpha beta gamma", section="S")
  index = DualVectorIndex(HashEmbedder(), lexical_weight=0.0)
  index.add_blocks([block])
  results = index.search("alpha beta gamma", top_k=1)
  # With lexical disabled, the reported score is exactly the backend's
  # cosine score (no lexical blend added).
  backend_score = index.backend.search("alpha beta gamma", top_k=1)[0][1]
  assert results[0].score == pytest.approx(backend_score)


# ------------------------------------------- pipeline staged ingestion
def test_pipeline_separate_parse_chunk_index_stages():
  pipeline = TableRAGPipeline(
    embedder=HashEmbedder(), enable_compute=True, max_text_words=40
  )
  blocks = pipeline.parse(DOC2)
  assert len(blocks) > 0

  chunked = pipeline.chunk(blocks)
  # The MEMO section (~60 words) must have been split by the 40-word budget.
  memo_parts = [b for b in chunked if "MEMO" in b.section and "(part" in b.section]
  assert len(memo_parts) >= 2

  original_tables = [b for b in blocks if isinstance(b, TableBlock)]
  pipeline.index_blocks(chunked, sandbox_tables=original_tables)
  assert len(pipeline.index) == len(chunked)
  assert len(pipeline.sandbox) == len(original_tables)

  results = pipeline.index.search("net_rev_eur DE-442", top_k=3)
  assert any("EUROPE" in r.block.section for r in results)


def test_pipeline_vectorstore_shorthand():
  pytest.importorskip("langchain_core")
  from langchain_core.vectorstores import InMemoryVectorStore

  from tests.test_langchain_integration import FakeLCEmbeddings

  store = InMemoryVectorStore(embedding=FakeLCEmbeddings())
  pipeline = TableRAGPipeline(vectorstore=store, enable_compute=False)
  assert isinstance(pipeline.index.backend, LangChainVectorStoreBackend)

  pipeline.ingest(DOC2)
  assert len(store.store) == len(pipeline.index)

  results = pipeline.index.search(
    "WB-8841 A1-04 wireless buds oh reserved avail", top_k=3
  )
  assert any("PRODUCT MOVEMENT" in r.block.section for r in results)
