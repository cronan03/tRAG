"""Build searchable summaries for blocks (dual-vector indexing).

The summary is what gets embedded for retrieval; the raw block stays in the
DocStore. Anything satisfying the Summarizer protocol (`(Block) -> str`)
works:

- DeterministicSummarizer  schema + distinct values + notes (default, free)
- LLMSummarizer            narrative summary via any Generator
- any callable / custom class with the same signature

Example — custom summarizer:

    def my_summarizer(block):
        if block.kind == "table":
            return f"Sales grid: {', '.join(block.headers)}"
        return block.content[:500]

    pipe = TableRAGPipeline(generator=..., embedder=..., summarizer=my_summarizer)
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from tablerag.models import Block, TableBlock, TextBlock

# Defaults for the deterministic path (also module-level aliases for
# backward compatibility with older imports).
DEFAULT_MAX_DISTINCT_VALUES = 24
DEFAULT_MAX_TEXT_CHARS = 1200
MAX_DISTINCT_VALUES = DEFAULT_MAX_DISTINCT_VALUES
MAX_TEXT_CHARS = DEFAULT_MAX_TEXT_CHARS

DEFAULT_LLM_SUMMARY_PROMPT = (
  "You are summarizing a document block for semantic search retrieval. "
  "Write a detailed narrative summary of what the block contains so that "
  "a user query about its topic or metrics would match it. Mention the "
  "section title, key columns/entities, and any correction notes. "
  "Do not invent values that are not present.\n\n"
  "Block:\n{block}\n\nSummary:"
)


@runtime_checkable
class Summarizer(Protocol):
  """Minimal contract: turn a Block into a searchable summary string."""

  def __call__(self, block: Block) -> str: ...


# Type alias for documentation / annotations.
SummarizerFn = Callable[[Block], str]


class DeterministicSummarizer:
  """Schema + distinct cell values + context notes. Zero LLM calls.

  Args:
    max_distinct_values: max unique values listed per table column
      (default 24). Raise for wide identifier-heavy tables.
    max_text_chars: max characters kept for prose TextBlocks
      (default 1200).
  """

  def __init__(
    self,
    max_distinct_values: int = DEFAULT_MAX_DISTINCT_VALUES,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
  ) -> None:
    self.max_distinct_values = max_distinct_values
    self.max_text_chars = max_text_chars

  def _column_values_line(self, table: TableBlock, col_index: int) -> str:
    values = []
    seen = set()
    for row in table.rows:
      value = row[col_index]
      if value and value not in seen:
        seen.add(value)
        values.append(value)
      if len(values) >= self.max_distinct_values:
        break
    return ", ".join(values)

  def summarize_table(self, table: TableBlock) -> str:
    """Build a searchable summary for a TableBlock."""
    lines = [
      f"Table from section: {table.section}.",
      f"Columns: {', '.join(table.headers)}.",
      f"{table.num_rows} rows.",
    ]
    for i, header in enumerate(table.headers):
      values = self._column_values_line(table, i)
      if values:
        lines.append(f"{header} values: {values}.")
    if table.context_notes:
      lines.append(f"Notes: {table.context_notes}")
    return "\n".join(lines)

  def summarize_text(self, text: TextBlock) -> str:
    """Build a searchable summary for a TextBlock."""
    content = text.content[: self.max_text_chars]
    if text.section and text.section not in content:
      return f"Section: {text.section}.\n{content}"
    return content

  def __call__(self, block: Block) -> str:
    if isinstance(block, TableBlock):
      return self.summarize_table(block)
    return self.summarize_text(block)


# Module-level default instance (shared caps).
_DEFAULT = DeterministicSummarizer()


def summarize_table(table: TableBlock) -> str:
  """Convenience: summarize a table with default caps."""
  return _DEFAULT.summarize_table(table)


def summarize_text(text: TextBlock) -> str:
  """Convenience: summarize prose with default caps."""
  return _DEFAULT.summarize_text(text)


def summarize_block(block: Block) -> str:
  """Default summarizer entry point (DeterministicSummarizer with default caps).

  Used as the DualVectorIndex / TableRAGPipeline default when no
  summarizer= is passed.
  """
  return _DEFAULT(block)


class LLMSummarizer:
  """Narrative table/text summary via any Generator (classic dual-vector).

  Costs one LLM call per block at ingest. Prefer the deterministic default
  unless fuzzy / narrative queries need richer summaries.

  Args:
    generator: any Generator (`generate(prompt) -> str`).
    prompt_template: format string with a `{block}` slot. Defaults to
      DEFAULT_LLM_SUMMARY_PROMPT.
  """

  def __init__(
    self,
    generator,
    prompt_template: str | None = None,
  ) -> None:
    self.generator = generator
    self.prompt_template = prompt_template or DEFAULT_LLM_SUMMARY_PROMPT

  def _render_block(self, block: Block) -> str:
    if isinstance(block, TableBlock):
      parts = [f"### {block.section}", block.markdown]
      if block.context_notes:
        parts.append(f"Notes: {block.context_notes}")
      return "\n".join(parts)
    return f"### {block.section}\n{block.content}"

  def __call__(self, block: Block) -> str:
    prompt = self.prompt_template.format(block=self._render_block(block))
    return (self.generator.generate(prompt) or "").strip()
