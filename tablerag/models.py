"""Core data models for tablerag (pure Python, no framework dependencies)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def new_block_id() -> str:
  return str(uuid.uuid4())


@dataclass
class TextBlock:
  """A prose block (narrative, memos, irregular table-ish data)."""

  content: str
  section: str = ""
  doc_id: str = field(default_factory=new_block_id)
  source: str = ""

  @property
  def kind(self) -> str:
    return "text"


@dataclass
class TableBlock:
  """A table normalized to canonical markdown.

  `markdown` always contains a well-formed pipe table:
  header row, separator row, data rows. `headers` and `rows` hold the
  parsed grid for downstream SQL loading.
  """

  markdown: str
  headers: list[str]
  rows: list[list[str]]
  section: str = ""
  doc_id: str = field(default_factory=new_block_id)
  source: str = ""
  source_format: str = "markdown"  # markdown | tsv | csv | section-grid
  context_notes: str = ""  # nearby prose tied to this table (e.g. corrections)

  @property
  def kind(self) -> str:
    return "table"

  @property
  def num_rows(self) -> int:
    return len(self.rows)

  @property
  def num_cols(self) -> int:
    return len(self.headers)


Block = TextBlock | TableBlock


@dataclass
class RetrievedBlock:
  """A block returned from the index with its similarity score."""

  block: Block
  score: float


@dataclass
class QueryResult:
  """Structured result of a pipeline query."""

  query: str
  answer: str
  route: str  # lookup | compute | hybrid
  retrieved: list[RetrievedBlock] = field(default_factory=list)
  sql: str | None = None
  sql_result: list[dict] | None = None
  prompt: str = ""
  model: str = ""
