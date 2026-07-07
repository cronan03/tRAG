"""Unit tests for section splitting, table detection, and normalization."""

from pathlib import Path

import pytest

from tablerag.models import TableBlock, TextBlock
from tablerag.parse import parse_document, split_sections, try_parse_table

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------- sections
def test_banner_sections_document2():
  text = (DATA_DIR / "document2.txt").read_text(encoding="utf-8")
  sections = split_sections(text)
  titles = [s.title for s in sections]

  assert any("MEMO" in t for t in titles)
  assert any("NORTH AMERICA" in t for t in titles)
  assert any("EUROPE" in t for t in titles)
  assert any("DEFINITIONS" in t for t in titles)
  # Preamble before the first banner is preserved.
  assert titles[0] == "(preamble)"


def test_hr_sections_tables_context():
  text = (DATA_DIR / "tables_context.txt").read_text(encoding="utf-8")
  sections = split_sections(text)
  assert len(sections) == 5
  assert any("orders" in s.title for s in sections)


def test_fallback_single_section():
  sections = split_sections("just a plain paragraph")
  assert len(sections) == 1
  assert sections[0].body == "just a plain paragraph"


# ------------------------------------------------------------ table parsing
def test_parse_pipe_table_with_borders():
  lines = [
    "| loc | sku | oh |",
    "| A1-04 | WB-8841 | 4200 |",
    "| B2-11 | CH-902 | 85 |",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.headers == ["loc", "sku", "oh"]
  assert table.rows == [["A1-04", "WB-8841", "4200"], ["B2-11", "CH-902", "85"]]
  assert table.source_format == "pipe"


def test_parse_pipe_table_without_borders():
  lines = [
    "wk_end | net_rev_eur | store",
    "2025-06-07 | 412880.30 | FR-901",
    "2025-06-14 | 428110.00 | FR-901",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.headers == ["wk_end", "net_rev_eur", "store"]
  assert table.num_rows == 2


def test_parse_compact_pipe_table():
  lines = [
    "emp_id|name|dept",
    "E1001|Alice Johnson|ENG",
    "E1002|Bob Smith|SALES",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.headers == ["emp_id", "name", "dept"]
  assert table.rows[0] == ["E1001", "Alice Johnson", "ENG"]


def test_parse_tsv_table():
  lines = [
    "store_id\tregion\tnet_rev_usd",
    "S0142\tNA-EAST\t612940.55",
    "S0201\tNA-WEST\t891520.00",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.source_format == "tsv"
  assert table.headers == ["store_id", "region", "net_rev_usd"]


def test_parse_csv_table_with_trailing_empty_cell():
  lines = [
    "store,week,rev_usd,units,refunds",
    "JP-77,2025-06-07,188420.00,1422,31",
    "SG-05,2025-06-07,55220.00,418,",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.source_format == "csv"
  assert table.rows[-1] == ["SG-05", "2025-06-07", "55220.00", "418", ""]


def test_markdown_separator_rows_are_dropped():
  lines = [
    "| Quarter | Revenue |",
    "| :--- | :--- |",
    "| Q1 | 12.5 |",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.num_rows == 1
  assert table.rows[0] == ["Q1", "12.5"]


def test_prose_is_not_a_table():
  lines = [
    "Revenue looked soft in APAC but NA held up, according to ops.",
    "Returns spiked after firmware v3.2.1; see ticket cluster below.",
  ]
  assert try_parse_table(lines) is None


def test_key_value_lines_are_not_a_table():
  lines = [
    "order_id=8829101 ts=2025-06-27T09:14:22Z sku=WB-8841 qty=2",
    "order_id=8829102 ts=2025-06-27T09:31:05Z sku=CH-902 qty=1",
  ]
  assert try_parse_table(lines) is None


def test_canonical_markdown_output():
  lines = [
    "a\tb",
    "1\t2",
  ]
  table = try_parse_table(lines)
  assert table is not None
  assert table.markdown == "| a | b |\n| --- | --- |\n| 1 | 2 |"


# ---------------------------------------------------------- full document
@pytest.fixture(scope="module")
def doc2_blocks():
  text = (DATA_DIR / "document2.txt").read_text(encoding="utf-8")
  return parse_document(text, source="document2.txt")


def test_document2_finds_core_tables(doc2_blocks):
  tables = [b for b in doc2_blocks if isinstance(b, TableBlock)]
  sections = {t.section for t in tables}

  assert any("NORTH AMERICA" in s for s in sections)
  assert any("EUROPE" in s for s in sections)
  assert any("APAC" in s for s in sections)
  assert any("PRODUCT MOVEMENT" in s for s in sections)
  assert any("HR & PAYROLL" in s for s in sections)


def test_document2_warehouse_table_shape(doc2_blocks):
  tables = [b for b in doc2_blocks if isinstance(b, TableBlock)]
  warehouse = next(t for t in tables if "PRODUCT MOVEMENT" in t.section)
  assert warehouse.headers == [
    "loc",
    "sku",
    "description",
    "oh",
    "reserved",
    "avail",
    "last_recv",
  ]
  assert warehouse.num_rows == 7


def test_document2_correction_note_travels_with_eu_table(doc2_blocks):
  tables = [b for b in doc2_blocks if isinstance(b, TableBlock)]
  eu = next(t for t in tables if "EUROPE" in t.section)
  assert "391005.15" in eu.context_notes


def test_document2_prose_sections_stay_text(doc2_blocks):
  texts = [b for b in doc2_blocks if isinstance(b, TextBlock)]
  sections = {t.section for t in texts}
  assert any("MEMO" in s for s in sections)
  assert any("MARKETING" in s for s in sections)
  assert any("DEFINITIONS" in s for s in sections)


def test_document2_shopify_jsonl_is_text_not_table(doc2_blocks):
  shopify_tables = [
    b for b in doc2_blocks if isinstance(b, TableBlock) and "SHOPIFY" in b.section
  ]
  assert shopify_tables == []
  shopify_texts = [
    b for b in doc2_blocks if isinstance(b, TextBlock) and "SHOPIFY" in b.section
  ]
  assert len(shopify_texts) >= 1
