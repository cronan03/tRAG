"""Turn raw document text into TextBlock / TableBlock lists."""

from __future__ import annotations

from tablerag.models import Block, TextBlock
from tablerag.parse.sections import split_sections
from tablerag.parse.tables import try_parse_table


def _split_paragraphs(body: str) -> list[list[str]]:
  """Group section body lines into contiguous non-blank paragraphs."""
  paragraphs: list[list[str]] = []
  current: list[str] = []
  for line in body.splitlines():
    if line.strip():
      current.append(line)
    elif current:
      paragraphs.append(current)
      current = []
  if current:
    paragraphs.append(current)
  return paragraphs


def parse_document(text: str, *, source: str = "") -> list[Block]:
  """Parse heterogeneous text into TextBlocks and TableBlocks.

  - Sections come from banner / horizontal-rule delimiters.
  - Within a section, each blank-line-separated paragraph is tested as a
    table (pipe / TSV / CSV). Failures stay prose.
  - Prose paragraphs in a section that also contains a table are attached to
    that table's `context_notes` (so e.g. correction footnotes travel with
    the table) AND kept as standalone TextBlocks for retrieval.
  """
  blocks: list[Block] = []

  for section in split_sections(text):
    section_blocks: list[Block] = []
    for paragraph in _split_paragraphs(section.body):
      table = try_parse_table(paragraph, section=section.title, source=source)
      if table:
        section_blocks.append(table)
      else:
        section_blocks.append(
          TextBlock(
            content="\n".join(paragraph).strip(),
            section=section.title,
            source=source,
          )
        )

    tables = [b for b in section_blocks if b.kind == "table"]
    texts = [b for b in section_blocks if b.kind == "text"]
    if tables and texts:
      notes = "\n\n".join(t.content for t in texts)
      for table in tables:
        table.context_notes = notes

    blocks.extend(section_blocks)

  return blocks
