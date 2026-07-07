"""Evaluation runner: ingest contexts, score retrieval, optionally score
generated answers.

Retrieval scoring is free (embedding calls only, or fully offline with
HashEmbedder). Generation scoring calls the LLM once per sample and is
opt-in via generate=True.
"""

from __future__ import annotations

import logging
import time

from tablerag.evals.metrics import answer_match, mrr_at_k, recall_at_k
from tablerag.evals.models import EvalReport, EvalResult, EvalSample
from tablerag.parse import parse_document
from tablerag.pipeline import TableRAGPipeline

logger = logging.getLogger("tablerag")


def _hit_rank(sample: EvalSample, retrieved_blocks) -> int | None:
  """1-based rank of the first retrieved block matching a golden marker."""
  for rank, item in enumerate(retrieved_blocks, start=1):
    block = item.block
    if sample.golden_sections:
      section = block.section.lower()
      if any(g.lower() in section for g in sample.golden_sections):
        return rank
      continue  # sections are the stricter marker when provided
    if sample.golden_sources and block.source in sample.golden_sources:
      return rank
  return None


class Evaluator:
  def __init__(
    self,
    pipeline: TableRAGPipeline,
    top_k: int = 3,
    generate: bool = False,
    rel_tol: float = 0.01,
    delay_sec: float = 0.0,
  ) -> None:
    """
    Args:
      pipeline: a configured TableRAGPipeline (its embedder/backend/chunking
        settings are what get evaluated).
      top_k: retrieval depth for Recall@k / MRR@k.
      generate: also run query() per sample and score answers (LLM calls).
      rel_tol: numeric tolerance for answer matching.
      delay_sec: sleep between generation calls (rate-limit friendliness).
    """
    self.pipeline = pipeline
    self.top_k = top_k
    self.generate = generate
    self.rel_tol = rel_tol
    self.delay_sec = delay_sec

  def ingest_contexts(self, contexts: dict[str, str]) -> None:
    """Ingest each context with source=context_id (golden_sources match it)."""
    from tablerag.models import TableBlock

    for context_id, text in contexts.items():
      blocks = parse_document(text, source=context_id)
      originals = [b for b in blocks if isinstance(b, TableBlock)]
      chunked = self.pipeline.chunk(blocks)
      self.pipeline.index_blocks(chunked, sandbox_tables=originals)
    logger.info(
      "Evaluator: ingested %d contexts (%d indexed blocks)",
      len(contexts),
      len(self.pipeline.index),
    )

  def evaluate_sample(self, sample: EvalSample) -> EvalResult:
    retrieved = self.pipeline.index.search(sample.question, top_k=self.top_k)
    result = EvalResult(
      sample_id=sample.id,
      question=sample.question,
      retrieved=[
        f"{r.block.kind}:{r.block.source}:{r.block.section[:50]}"
        for r in retrieved
      ],
      hit_rank=_hit_rank(sample, retrieved),
    )

    if self.generate:
      try:
        query_result = self.pipeline.query(sample.question, top_k=self.top_k)
        result.prediction = query_result.answer
        result.route = query_result.route
        result.sql = query_result.sql
        result.answer_correct = answer_match(
          query_result.answer,
          sample.answers,
          require_all=sample.require_all,
          rel_tol=self.rel_tol,
        )
      except Exception as exc:
        result.error = str(exc)
      if self.delay_sec:
        time.sleep(self.delay_sec)

    return result

  def run(
    self,
    samples: list[EvalSample],
    contexts: dict[str, str],
    dataset_name: str = "",
  ) -> EvalReport:
    self.ingest_contexts(contexts)

    results = []
    for i, sample in enumerate(samples, start=1):
      logger.info("[eval %d/%d] %s", i, len(samples), sample.question[:70])
      results.append(self.evaluate_sample(sample))

    hit_ranks = [r.hit_rank for r in results]
    generated = [r for r in results if r.answer_correct is not None]
    return EvalReport(
      dataset=dataset_name or (samples[0].dataset if samples else ""),
      top_k=self.top_k,
      num_samples=len(samples),
      recall_at_k=recall_at_k(hit_ranks, self.top_k),
      mrr_at_k=mrr_at_k(hit_ranks, self.top_k),
      num_generated=len(generated),
      exact_match=(
        sum(1 for r in generated if r.answer_correct) / len(generated)
        if generated
        else None
      ),
      results=results,
    )


def format_report(report: EvalReport) -> str:
  lines = [
    "=" * 72,
    f"Dataset: {report.dataset}   samples: {report.num_samples}   k={report.top_k}",
    f"Retrieval  Recall@{report.top_k}: {report.recall_at_k:.1%}"
    f"   MRR@{report.top_k}: {report.mrr_at_k:.3f}",
  ]
  if report.exact_match is not None:
    lines.append(
      f"Generation Exact Match: {report.exact_match:.1%}"
      f" ({report.num_generated} generated)"
    )
  else:
    lines.append("Generation: skipped (retrieval-only run)")
  lines.append("-" * 72)
  for r in report.results:
    hit = f"hit@{r.hit_rank}" if r.hit_rank else "MISS "
    answer = ""
    if r.answer_correct is not None:
      answer = "  answer=OK" if r.answer_correct else "  answer=WRONG"
    if r.error:
      answer = f"  ERROR: {r.error[:40]}"
    lines.append(f"[{hit}]{answer}  {r.question[:80]}")
  lines.append("=" * 72)
  return "\n".join(lines)
