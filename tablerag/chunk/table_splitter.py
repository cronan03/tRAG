"""Header-aware table splitting.

Large tables are sliced into row groups, but every slice re-injects the
column headers and keeps the section title + context notes, so no slice is
ever a floating grid of numbers.
"""

from __future__ import annotations

from tablerag.models import TableBlock
from tablerag.parse.tables import to_canonical_markdown

DEFAULT_MAX_ROWS = 50


def split_table_block(
  block: TableBlock,
  *,
  max_rows: int = DEFAULT_MAX_ROWS,
  overlap_rows: int = 0,
  max_words: int | None = None,
) -> list[TableBlock]:
  """Split a TableBlock into row-group slices with headers re-injected.

  Tables within the budget are returned unchanged (single element).

  Args:
    max_rows: hard row cap per slice (default 50).
    overlap_rows: rows repeated between consecutive slices (default 0).
    max_words: optional word budget per slice; converted to a row cap
      using the table's average words-per-row. The stricter of max_rows
      and the derived cap wins.
  """
  if max_words is not None and block.num_rows > 0:
    total_words = sum(len(cell.split()) for row in block.rows for cell in row)
    words_per_row = max(1, total_words // block.num_rows)
    header_words = sum(len(h.split()) for h in block.headers)
    budget = max(1, (max_words - header_words) // words_per_row)
    max_rows = min(max_rows, budget)

  if block.num_rows <= max_rows:
    return [block]

  step = max_rows - overlap_rows
  if step <= 0:
    raise ValueError("overlap_rows must be smaller than max_rows")

  slices: list[TableBlock] = []
  total = block.num_rows
  for start in range(0, total, step):
    rows = block.rows[start : start + max_rows]
    if not rows:
      break
    part = len(slices) + 1
    slices.append(
      TableBlock(
        markdown=to_canonical_markdown(block.headers, rows),
        headers=list(block.headers),
        rows=rows,
        section=f"{block.section} (rows {start + 1}-{start + len(rows)})",
        source=block.source,
        source_format=block.source_format,
        context_notes=block.context_notes,
      )
    )
    if start + max_rows >= total:
      break

  return slices
