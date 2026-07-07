"""Unit tests for the header-aware table splitter."""

from tablerag.chunk import split_table_block
from tablerag.models import TableBlock
from tablerag.parse.tables import to_canonical_markdown


def make_table(num_rows: int) -> TableBlock:
  headers = ["id", "value"]
  rows = [[str(i), str(i * 10)] for i in range(num_rows)]
  return TableBlock(
    markdown=to_canonical_markdown(headers, rows),
    headers=headers,
    rows=rows,
    section="BIG TABLE",
    context_notes="important footnote",
  )


def test_small_table_not_split():
  table = make_table(10)
  assert split_table_block(table, max_rows=50) == [table]


def test_large_table_split_with_headers_on_every_slice():
  table = make_table(120)
  slices = split_table_block(table, max_rows=50)

  assert len(slices) == 3
  for s in slices:
    assert s.headers == ["id", "value"]
    assert s.markdown.startswith("| id | value |")
    assert s.context_notes == "important footnote"
    assert "BIG TABLE" in s.section

  assert sum(s.num_rows for s in slices) == 120
  assert slices[0].rows[0] == ["0", "0"]
  assert slices[-1].rows[-1] == ["119", "1190"]


def test_split_with_overlap():
  table = make_table(100)
  slices = split_table_block(table, max_rows=50, overlap_rows=10)
  assert slices[1].rows[0] == table.rows[40]
