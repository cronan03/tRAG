"""Offline tests for the evaluation harness (HashEmbedder, no network)."""

import pytest

from tablerag.evals.loaders import (
  load_doc2,
  t2_records_to_eval,
  wtq_records_to_eval,
)
from tablerag.evals.metrics import (
  answer_match,
  mrr_at_k,
  numbers_in,
  parse_number,
  recall_at_k,
)
from tablerag.evals.runner import Evaluator
from tablerag.index.embedder import HashEmbedder
from tablerag.pipeline import TableRAGPipeline


# ------------------------------------------------------------- metrics
def test_answer_match_text_containment():
  assert answer_match("The CH-902 family is low.", ["CH-902"])
  assert not answer_match("The USB-H3 hub.", ["CH-902"])


def test_answer_match_numeric_tolerance():
  assert answer_match("about 1,965,043.30 USD", ["1965043.30"])
  assert answer_match("roughly 1,965,000", ["1965043.30"])  # within 1%
  assert not answer_match("500000", ["1965043.30"])


def test_answer_match_percent_vs_fraction():
  assert answer_match("the rate is 3.53%", ["3.53"])
  assert answer_match("0.0353", ["3.53"])  # fraction form


def test_answer_match_require_all():
  assert answer_match("WMS 3310 and Shopify 9999", ["3310", "9999"], require_all=True)
  assert not answer_match("only 3310 here", ["3310", "9999"], require_all=True)


def test_answer_match_k_suffix():
  assert answer_match("about $42k", ["42000"])
  assert answer_match("$42,000 attributable", ["42k"])


def test_parse_and_numbers():
  assert parse_number("1,965,043.30") == pytest.approx(1965043.30)
  assert parse_number("42k") == pytest.approx(42000)
  assert parse_number("not a number") is None
  assert numbers_in("a 12 and 3.5 and 1,000") == [12.0, 3.5, 1000.0]
  assert numbers_in("about $42k and 1.5m") == [42000.0, 1500000.0]


def test_recall_and_mrr():
  ranks = [1, 2, None, 4]
  assert recall_at_k(ranks, 3) == pytest.approx(0.5)  # ranks 1,2 within 3
  assert mrr_at_k(ranks, 3) == pytest.approx((1 / 1 + 1 / 2) / 4)
  assert recall_at_k([], 3) == 0.0


# ---------------------------------------------------------- doc2 loader
def test_load_doc2_shape():
  samples, contexts = load_doc2()
  assert len(samples) == 10
  assert "document2.txt" in contexts
  assert all(s.golden_sources == ["document2.txt"] for s in samples)
  assert all(s.golden_sections for s in samples)


# ----------------------------------------------- external loader converters
def test_wtq_records_to_eval():
  records = [
    {
      "id": "nt-1",
      "question": "Which nation won the most gold medals?",
      "answers": ["France"],
      "table": {
        "name": "csv/204-csv/1.csv",
        "header": ["nation", "gold"],
        "rows": [["France", "10"], ["Spain", "7"]],
      },
    }
  ]
  samples, contexts = wtq_records_to_eval(records)
  assert samples[0].golden_sources == ["csv/204-csv/1.csv"]
  assert samples[0].dataset == "wtq"
  assert "| nation | gold |" in contexts["csv/204-csv/1.csv"]


def test_t2_records_to_eval():
  records = [
    {
      "id": "finqa-1",
      "context_id": "AAL/2014/page_1.pdf",
      "question": "What was the percentage change in revenue?",
      "program_answer": "14.1",
      "original_answer": "14.1%",
      "context": "Revenue rose. | year | rev |\n| 2013 | 100 |\n| 2014 | 114 |",
    }
  ]
  samples, contexts = t2_records_to_eval(records)
  assert samples[0].answers == ["14.1", "14.1%"]
  assert samples[0].golden_sources == ["AAL/2014/page_1.pdf"]
  assert "AAL/2014/page_1.pdf" in contexts


# ------------------------------------------------- end-to-end retrieval eval
def test_evaluator_retrieval_only_on_doc2():
  samples, contexts = load_doc2()
  pipeline = TableRAGPipeline(embedder=HashEmbedder(), enable_compute=False)
  report = Evaluator(pipeline, top_k=3, generate=False).run(
    samples, contexts, dataset_name="doc2"
  )

  assert report.num_samples == 10
  assert report.exact_match is None  # generation skipped
  assert 0.0 <= report.recall_at_k <= 1.0
  # Hash embedder is weak, but hybrid lexical scoring should still land
  # several golden sections in the top 3.
  assert report.recall_at_k >= 0.5
  assert len(report.results) == 10


def test_evaluator_hit_rank_uses_sections():
  samples, contexts = load_doc2()
  eu = next(s for s in samples if s.id == "doc2-01")
  pipeline = TableRAGPipeline(embedder=HashEmbedder(), enable_compute=False)
  ev = Evaluator(pipeline, top_k=5, generate=False)
  ev.ingest_contexts(contexts)
  result = ev.evaluate_sample(eu)
  # EUROPE section should be retrievable for the DE-442 question.
  assert result.hit_rank is not None
