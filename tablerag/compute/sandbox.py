"""Ephemeral in-memory DuckDB sandbox for parsed tables.

Every TableBlock is loaded as a typed SQL table (ints, doubles, dates
inferred from cell values; empty cells become NULL). The sandbox is
read-only from the LLM's perspective: only single SELECT statements are
executed.
"""

from __future__ import annotations

import logging
import re

from tablerag.models import TableBlock

logger = logging.getLogger("tablerag")

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IDENT_RE = re.compile(r"[^a-z0-9_]+")

FORBIDDEN_SQL = re.compile(
  r"\b(insert|update|delete|drop|create|alter|attach|copy|install|load|pragma|set)\b",
  re.IGNORECASE,
)


def sanitize_identifier(name: str, fallback: str = "col") -> str:
  ident = _IDENT_RE.sub("_", name.strip().lower()).strip("_")
  if not ident or ident[0].isdigit():
    ident = f"{fallback}_{ident}" if ident else fallback
  return ident[:60]


def infer_column_type(values: list[str]) -> str:
  non_empty = [v for v in values if v.strip()]
  if not non_empty:
    return "VARCHAR"
  if all(_INT_RE.match(v) for v in non_empty):
    return "BIGINT"
  if all(_INT_RE.match(v) or _FLOAT_RE.match(v) for v in non_empty):
    return "DOUBLE"
  if all(_DATE_RE.match(v) for v in non_empty):
    return "DATE"
  return "VARCHAR"


class TableSandbox:
  def __init__(self) -> None:
    import duckdb

    self._conn = duckdb.connect(":memory:")
    self._tables: dict[str, TableBlock] = {}  # sql_name -> source block

  def __len__(self) -> int:
    return len(self._tables)

  # -------------------------------------------------------------- load
  def _unique_name(self, base: str) -> str:
    name, i = base, 2
    while name in self._tables:
      name = f"{base}_{i}"
      i += 1
    return name

  def load_table(self, block: TableBlock) -> str:
    """Load a TableBlock as a typed DuckDB table; returns the SQL name."""
    # Section titles are often "TOPIC — long description (caveats)"; the
    # topic alone makes the friendliest SQL name.
    title = re.split(r"[—\-(]", block.section, maxsplit=1)[0]
    base = sanitize_identifier(title, fallback="tbl") or "tbl"
    name = self._unique_name(base)

    columns = [
      sanitize_identifier(h, fallback=f"col{i}")
      for i, h in enumerate(block.headers)
    ]
    # De-duplicate column names.
    seen: dict[str, int] = {}
    for i, col in enumerate(columns):
      if col in seen:
        seen[col] += 1
        columns[i] = f"{col}_{seen[col]}"
      else:
        seen[col] = 1

    types = [
      infer_column_type([row[i] for row in block.rows])
      for i in range(len(columns))
    ]

    col_defs = ", ".join(f'"{c}" {t}' for c, t in zip(columns, types))
    self._conn.execute(f'CREATE TABLE "{name}" ({col_defs})')

    placeholders = ", ".join("?" for _ in columns)
    insert = f'INSERT INTO "{name}" VALUES ({placeholders})'
    for row in block.rows:
      values = [cell.strip() if cell.strip() else None for cell in row]
      self._conn.execute(insert, values)

    self._tables[name] = block
    logger.info(
      "Sandbox: loaded '%s' (%d rows, %d cols) from section %r",
      name,
      block.num_rows,
      len(columns),
      block.section,
    )
    return name

  def load_tables(self, blocks: list[TableBlock]) -> list[str]:
    return [self.load_table(b) for b in blocks]

  # ------------------------------------------------------------- schema
  def schema_description(self) -> str:
    """Human/LLM-readable schema with column types and sample rows."""
    parts = []
    for name, block in self._tables.items():
      info = self._conn.execute(f'DESCRIBE "{name}"').fetchall()
      cols = ", ".join(f"{row[0]} ({row[1]})" for row in info)
      sample = self._conn.execute(f'SELECT * FROM "{name}" LIMIT 2').fetchall()
      parts.append(
        f"Table: {name}\n"
        f"  Source section: {block.section}\n"
        f"  Columns: {cols}\n"
        f"  Sample rows: {sample}"
        + (f"\n  Notes: {block.context_notes}" if block.context_notes else "")
      )
    return "\n\n".join(parts)

  # ------------------------------------------------------------ execute
  def execute(self, sql: str) -> list[dict]:
    """Execute a single SELECT statement and return rows as dicts."""
    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
      raise ValueError("Only a single SQL statement is allowed.")
    if not cleaned.lower().lstrip("( ").startswith(("select", "with")):
      raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN_SQL.search(cleaned):
      raise ValueError("Statement contains forbidden SQL keywords.")

    logger.info("Sandbox SQL: %s", cleaned)
    cursor = self._conn.execute(cleaned)
    column_names = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(column_names, row)) for row in rows]

  def close(self) -> None:
    self._conn.close()
