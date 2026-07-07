"""Evaluation harness: dataset loaders + automated retrieval/answer scoring.

Datasets:
- doc2 (built-in, offline): the 10 stress queries over data/document2.txt
- WikiTableQuestions (HuggingFace: stanfordnlp/wikitablequestions)
- T2-RAGBench (HuggingFace: G4KMU/t2-ragbench)

External datasets need the optional `datasets` dependency:
    pip install tablerag[evals]

Run from the command line:
    python -m tablerag.evals doc2 --offline --no-generate
    python -m tablerag.evals wtq --sample-size 50
"""

from tablerag.evals.loaders import load_doc2, load_t2ragbench, load_wtq
from tablerag.evals.metrics import answer_match, mrr_at_k, recall_at_k
from tablerag.evals.models import EvalReport, EvalResult, EvalSample
from tablerag.evals.runner import Evaluator, format_report

__all__ = [
  "EvalSample",
  "EvalResult",
  "EvalReport",
  "Evaluator",
  "format_report",
  "load_doc2",
  "load_wtq",
  "load_t2ragbench",
  "answer_match",
  "recall_at_k",
  "mrr_at_k",
]
