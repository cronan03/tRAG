"""Offline tests for the dual-vector index using the hash embedder."""

from pathlib import Path

import pytest

from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import HashEmbedder
from tablerag.models import TableBlock
from tablerag.parse import parse_document

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def doc2_index():
  text = (DATA_DIR / "document2.txt").read_text(encoding="utf-8")
  blocks = parse_document(text, source="document2.txt")
  index = DualVectorIndex(HashEmbedder())
  index.add_blocks(blocks)
  return index


def test_index_returns_raw_blocks_not_summaries(doc2_index):
  results = doc2_index.search("warehouse stock on hand for wireless buds", top_k=3)
  assert len(results) == 3
  # Raw payload check: a retrieved table block still has its full markdown.
  table_hits = [r for r in results if isinstance(r.block, TableBlock)]
  if table_hits:
    assert "|" in table_hits[0].block.markdown


def test_warehouse_query_retrieves_warehouse_table(doc2_index):
  results = doc2_index.search(
    "average stock quantity oh avail for products in the warehouse", top_k=3
  )
  sections = [r.block.section for r in results]
  assert any("PRODUCT MOVEMENT" in s for s in sections)


def test_na_revenue_query_retrieves_na_table(doc2_index):
  results = doc2_index.search(
    "total net_rev_usd across NA stores week 2025-06-14", top_k=3
  )
  sections = [r.block.section for r in results]
  assert any("NORTH AMERICA" in s for s in sections)


def test_docstore_roundtrip(tmp_path, doc2_index):
  path = tmp_path / "store.json"
  doc2_index.docstore.save(path)

  from tablerag.index.docstore import DocStore

  loaded = DocStore.load(path)
  assert len(loaded) == len(doc2_index.docstore)
  original = doc2_index.docstore.all_blocks()[0]
  assert loaded.get(original.doc_id).section == original.section
