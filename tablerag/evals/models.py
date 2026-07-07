"""Data models for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalSample:
  """One question with its gold answer(s) and golden context markers.

  Retrieval is scored as a hit when a retrieved block matches
  `golden_sources` (exact match on block.source, i.e. the context_id the
  block was ingested under) or `golden_sections` (case-insensitive
  substring match on block.section).
  """

  id: str
  question: str
  answers: list[str]
  golden_sources: list[str] = field(default_factory=list)
  golden_sections: list[str] = field(default_factory=list)
  require_all: bool = False  # True: prediction must contain ALL gold answers
  dataset: str = ""


@dataclass
class EvalResult:
  sample_id: str
  question: str
  retrieved: list[str]  # "kind:source:section" per retrieved block
  hit_rank: int | None  # 1-based rank of first golden block, None = miss
  prediction: str | None = None  # None when generation is skipped
  answer_correct: bool | None = None
  route: str | None = None
  sql: str | None = None
  error: str | None = None


@dataclass
class EvalReport:
  dataset: str
  top_k: int
  num_samples: int
  recall_at_k: float
  mrr_at_k: float
  num_generated: int
  exact_match: float | None  # None when generation was skipped
  results: list[EvalResult] = field(default_factory=list)
