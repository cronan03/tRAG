"""Build searchable summaries for blocks.

Default mode is deterministic (schema + distinct values + context notes):
zero LLM calls, fully reproducible, and rich enough for semantic embedding
because it names the section, every column, and the entity values queries
mention. An LLM mode can be layered on later.
"""

from __future__ import annotations

from tablerag.models import Block, TableBlock, TextBlock

MAX_DISTINCT_VALUES = 24
MAX_TEXT_CHARS = 1200


def _column_values_line(table: TableBlock, col_index: int) -> str:
  values = []
  seen = set()
  for row in table.rows:
    value = row[col_index]
    if value and value not in seen:
      seen.add(value)
      values.append(value)
    if len(values) >= MAX_DISTINCT_VALUES:
      break
  return ", ".join(values)


def summarize_table(table: TableBlock) -> str:
  lines = [
    f"Table from section: {table.section}.",
    f"Columns: {', '.join(table.headers)}.",
    f"{table.num_rows} rows.",
  ]
  for i, header in enumerate(table.headers):
    values = _column_values_line(table, i)
    if values:
      lines.append(f"{header} values: {values}.")
  if table.context_notes:
    lines.append(f"Notes: {table.context_notes}")
  return "\n".join(lines)


def summarize_text(text: TextBlock) -> str:
  content = text.content[:MAX_TEXT_CHARS]
  if text.section and text.section not in content:
    return f"Section: {text.section}.\n{content}"
  return content


def summarize_block(block: Block) -> str:
  if isinstance(block, TableBlock):
    return summarize_table(block)
  return summarize_text(block)
