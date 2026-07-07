"""Detect table formats and normalize them to canonical markdown.

v1 contract (full retrieval + SQL support):
- Markdown / pipe-delimited rows  (`| a | b |` or `a | b` or `a|b`)
- TSV (tab-delimited)
- CSV (comma-delimited with header row)

Everything else stays prose (TextBlock, retrieval-only).
"""

from __future__ import annotations

import re

from tablerag.models import TableBlock

_SEPARATOR_ROW_RE = re.compile(r"^[\s|:\-]+$")
MIN_TABLE_LINES = 2  # header + at least one data row
MIN_COLUMNS = 2


def _is_separator_row(line: str) -> bool:
  return bool(_SEPARATOR_ROW_RE.match(line)) and "-" in line


def _split_pipe_row(line: str) -> list[str]:
  stripped = line.strip()
  if stripped.startswith("|"):
    stripped = stripped[1:]
  if stripped.endswith("|"):
    stripped = stripped[:-1]
  return [cell.strip() for cell in stripped.split("|")]


def _try_pipe(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
  data_lines = [ln for ln in lines if not _is_separator_row(ln)]
  if len(data_lines) < MIN_TABLE_LINES:
    return None
  if not all(ln.count("|") >= 1 for ln in data_lines):
    return None

  rows = [_split_pipe_row(ln) for ln in data_lines]
  width = len(rows[0])
  if width < MIN_COLUMNS or any(len(r) != width for r in rows):
    return None
  return rows[0], rows[1:]


def _try_tsv(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
  if len(lines) < MIN_TABLE_LINES:
    return None
  if not all("\t" in ln for ln in lines):
    return None

  rows = [[cell.strip() for cell in ln.split("\t")] for ln in lines]
  width = len(rows[0])
  if width < MIN_COLUMNS or any(len(r) != width for r in rows):
    return None
  return rows[0], rows[1:]


def _try_csv(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
  if len(lines) < MIN_TABLE_LINES:
    return None
  counts = [ln.count(",") for ln in lines]
  if counts[0] < 1 or any(c != counts[0] for c in counts):
    return None

  rows = [[cell.strip() for cell in ln.split(",")] for ln in lines]
  width = len(rows[0])
  if width < MIN_COLUMNS:
    return None
  # Header sanity: header cells should be short identifiers, not sentences.
  if any(len(cell) > 40 or " " in cell.strip() for cell in rows[0]):
    return None
  return rows[0], rows[1:]


def to_canonical_markdown(headers: list[str], rows: list[list[str]]) -> str:
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join("---" for _ in headers) + " |",
  ]
  lines.extend("| " + " | ".join(row) + " |" for row in rows)
  return "\n".join(lines)


def try_parse_table(
  lines: list[str],
  *,
  section: str = "",
  source: str = "",
) -> TableBlock | None:
  """Try to parse a contiguous group of non-empty lines as a table.

  Returns a TableBlock (normalized to canonical markdown) or None.
  """
  stripped = [ln.rstrip() for ln in lines if ln.strip()]
  if len(stripped) < MIN_TABLE_LINES:
    return None

  for parser, fmt in ((_try_pipe, "pipe"), (_try_tsv, "tsv"), (_try_csv, "csv")):
    parsed = parser(stripped)
    if parsed:
      headers, rows = parsed
      return TableBlock(
        markdown=to_canonical_markdown(headers, rows),
        headers=headers,
        rows=rows,
        section=section,
        source=source,
        source_format=fmt,
      )
  return None
