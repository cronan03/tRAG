"""Tests for the query classifier and DuckDB sandbox (no LLM calls)."""

from pathlib import Path

import pytest

from tablerag.compute.sandbox import TableSandbox, infer_column_type, sanitize_identifier
from tablerag.models import TableBlock
from tablerag.parse import parse_document
from tablerag.route import classify_query

DATA_DIR = Path(__file__).parent.parent / "data"


# ------------------------------------------------------------- classifier
@pytest.mark.parametrize(
  "query",
  [
    "What is the total net revenue in USD across all NA stores for 2025-06-14?",
    "What is the average stock_quantity for products in the warehouse?",
    "What is the refund rate for NA-WEST stores for the week ending 2025-06-14?",
    "How many support tickets mention firmware v3.2.1?",
    "Which product variants are below the reorder threshold?",
    "Which region had the highest growth?",
  ],
)
def test_compute_queries(query):
  assert classify_query(query) == "compute"


@pytest.mark.parametrize(
  "query",
  [
    "What is the corrected net_rev_eur for DE-442 on 2025-06-07?",
    "Who signed the Q4 operational summary footnote?",
    "What does the oh column mean?",
  ],
)
def test_lookup_queries(query):
  assert classify_query(query) == "lookup"


# ---------------------------------------------------------------- helpers
def test_sanitize_identifier():
  assert sanitize_identifier("NORTH AMERICA — weekly rollup") == "north_america_weekly_rollup"
  assert sanitize_identifier("net_rev_usd") == "net_rev_usd"
  assert sanitize_identifier("123abc") == "col_123abc"


def test_infer_column_type():
  assert infer_column_type(["1", "2", "3"]) == "BIGINT"
  assert infer_column_type(["1.5", "2", "3"]) == "DOUBLE"
  assert infer_column_type(["2025-06-07", "2025-06-14"]) == "DATE"
  assert infer_column_type(["S0142", "S0201"]) == "VARCHAR"
  assert infer_column_type(["418", ""]) == "BIGINT"  # empty -> NULL


# ---------------------------------------------------------------- sandbox
@pytest.fixture()
def doc2_sandbox():
  text = (DATA_DIR / "document2.txt").read_text(encoding="utf-8")
  blocks = parse_document(text, source="document2.txt")
  tables = [b for b in blocks if isinstance(b, TableBlock)]
  sandbox = TableSandbox()
  sandbox.load_tables(tables)
  yield sandbox
  sandbox.close()


def test_sandbox_loads_document2_tables(doc2_sandbox):
  schema = doc2_sandbox.schema_description()
  assert "north_america" in schema
  assert "net_rev_usd" in schema
  assert len(doc2_sandbox) >= 5


def test_na_total_revenue_query(doc2_sandbox):
  rows = doc2_sandbox.execute(
    "SELECT SUM(net_rev_usd) AS total FROM north_america "
    "WHERE wk_end = DATE '2025-06-14'"
  )
  assert rows[0]["total"] == pytest.approx(1965043.30)


def test_warehouse_average_oh(doc2_sandbox):
  rows = doc2_sandbox.execute(
    "SELECT AVG(oh) AS avg_oh FROM product_movement"
  )
  assert rows[0]["avg_oh"] == pytest.approx(851.857, abs=0.01)


def test_null_handling_in_truncated_apac_export(doc2_sandbox):
  rows = doc2_sandbox.execute(
    "SELECT refunds FROM apac WHERE store = 'SG-05' AND week = DATE '2025-06-07'"
  )
  assert rows[0]["refunds"] is None


def test_rejects_non_select(doc2_sandbox):
  with pytest.raises(ValueError):
    doc2_sandbox.execute("DROP TABLE north_america")
  with pytest.raises(ValueError):
    doc2_sandbox.execute("SELECT 1; SELECT 2")


def test_rejects_forbidden_keywords(doc2_sandbox):
  with pytest.raises(ValueError):
    doc2_sandbox.execute("SELECT * FROM north_america WHERE 1=1 OR (DELETE FROM apac)")
