"""Dataset loaders.

Each loader returns (samples, contexts):
- samples: list[EvalSample]
- contexts: dict[context_id -> document text]. The Evaluator ingests every
  context with source=context_id, which is what golden_sources match on.

doc2 is built-in and offline. WTQ and T2-RAGBench download via the optional
HuggingFace `datasets` dependency (pip install tablerag[evals]); their
record-to-sample conversion is factored into pure functions so it can be
tested without network access.
"""

from __future__ import annotations

from pathlib import Path

from tablerag.evals.models import EvalSample
from tablerag.parse.tables import to_canonical_markdown

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ------------------------------------------------------------------ doc2
def load_doc2() -> tuple[list[EvalSample], dict[str, str]]:
  """The 10 document2 stress queries with machine-checkable gold answers.

  Gold answers were hand-computed from the raw document (we authored it);
  numeric golds match within 1% relative tolerance, text golds by
  normalized containment.
  """
  contexts = {
    "document2.txt": (DATA_DIR / "document2.txt").read_text(encoding="utf-8")
  }

  samples = [
    EvalSample(
      id="doc2-01",
      question="What is the corrected net_rev_eur for DE-442 on 2025-06-07?",
      answers=["391005.15"],
      golden_sections=["EUROPE"],
    ),
    EvalSample(
      id="doc2-02",
      question="What is the average stock_quantity for products in the warehouse?",
      # No stock_quantity column exists; avg oh = 851.86, avg avail = 685.43.
      answers=["851.86", "685.43"],
      golden_sections=["PRODUCT MOVEMENT"],
    ),
    EvalSample(
      id="doc2-03",
      question=(
        "How many units of WB-8841 are available to sell according to "
        "Shopify vs WMS?"
      ),
      answers=["3310", "3295"],
      require_all=True,
      golden_sections=["INVENTORY RECONCILIATION", "PRODUCT MOVEMENT"],
    ),
    EvalSample(
      id="doc2-04",
      question=(
        "What is the refund rate for NA-WEST stores for the week ending "
        "2025-06-14?"
      ),
      # (189 + 19) / (4988 + 902) = 3.53% combined.
      answers=["3.53"],
      golden_sections=["NORTH AMERICA"],
    ),
    EvalSample(
      id="doc2-05",
      question=(
        "What is the total net revenue in USD across all NA stores for "
        "2025-06-14?"
      ),
      answers=["1965043.30"],
      golden_sections=["NORTH AMERICA"],
    ),
    EvalSample(
      id="doc2-06",
      question=(
        "How many support tickets mention firmware v3.2.1 and requested "
        "a refund?"
      ),
      # Correct behavior is refusing: per-ticket refund status is unknown
      # for the 37-ticket cluster. Accept an explicit 0 or a hedge.
      answers=["0", "don't know", "not specified", "unknown", "does not"],
      golden_sections=["CUSTOMER SUPPORT"],
    ),
    EvalSample(
      id="doc2-07",
      question=(
        "What is the attributable incremental revenue from campaign "
        "SummerGlow?"
      ),
      answers=["42000", "68000", "42k", "68k"],
      golden_sections=["MARKETING"],
    ),
    EvalSample(
      id="doc2-08",
      question="What is the average basket size in NA?",
      # Simple avg of avg_basket (all 8 rows) = 156.50; 2025-06-14 rows
      # only = 156.60. Both within 1% of 156.5.
      answers=["156.5"],
      golden_sections=["NORTH AMERICA"],
    ),
    EvalSample(
      id="doc2-09",
      question="Which product variants are below the reorder threshold?",
      answers=["CH-902"],
      golden_sections=["INVENTORY RECONCILIATION", "PRODUCT MOVEMENT"],
    ),
    EvalSample(
      id="doc2-10",
      question="What is the APAC refund count for SG-05 on 2025-06-07?",
      # Field is empty in the truncated export; correct answer is a refusal.
      answers=["null", "missing", "unknown", "not available", "empty", "don't know"],
      golden_sections=["APAC"],
    ),
  ]
  for sample in samples:
    sample.dataset = "doc2"
    sample.golden_sources = ["document2.txt"]
  return samples, contexts


# ------------------------------------------------------------------- WTQ
def wtq_records_to_eval(
  records: list[dict],
) -> tuple[list[EvalSample], dict[str, str]]:
  """Convert WikiTableQuestions records (HF schema: id, question, answers,
  table{header, rows, name}) into eval samples + a markdown table corpus.

  Every question's golden context is its own table; all tables together
  form the retrieval corpus, so the harness measures whether the right
  table is found among all of them.
  """
  samples: list[EvalSample] = []
  contexts: dict[str, str] = {}

  for record in records:
    table = record["table"]
    context_id = table.get("name") or f"table-{record['id']}"
    if context_id not in contexts:
      contexts[context_id] = to_canonical_markdown(
        list(table["header"]), [list(row) for row in table["rows"]]
      )
    samples.append(
      EvalSample(
        id=str(record["id"]),
        question=record["question"],
        answers=[str(a) for a in record["answers"]],
        golden_sources=[context_id],
        dataset="wtq",
      )
    )
  return samples, contexts


# Parquet-native mirror used when the official script-based repo can't be
# loaded (datasets >= 4.0 dropped dataset-script support). Same schema, but
# only "train"/"test" splits and a single "default" config.
_WTQ_MIRROR = "lighteval/wikitablequestions"


def load_wtq(
  sample_size: int = 50,
  split: str = "validation",
  config: str = "random-split-1",
) -> tuple[list[EvalSample], dict[str, str]]:
  """Load a WikiTableQuestions subset (needs `pip install tablerag[evals]`).

  Tries the official `stanfordnlp/wikitablequestions` first; if the installed
  `datasets` version rejects its loading script, falls back to the
  parquet-native `lighteval/wikitablequestions` mirror (splits: train/test).

  Args:
    sample_size: number of questions (their tables form the corpus).
    split: "train" | "validation" | "test".
    config: HF config, "random-split-1" .. "random-split-5" (official only).
  """
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise ImportError(
      "WikiTableQuestions requires the 'datasets' package: "
      "pip install tablerag[evals]"
    ) from exc

  try:
    dataset = load_dataset("stanfordnlp/wikitablequestions", config, split=split)
  except (RuntimeError, ValueError):
    # Script-based load unsupported, or config/split absent: use the mirror.
    mirror_split = "test" if split in ("validation", "test") else "train"
    dataset = load_dataset(_WTQ_MIRROR, split=mirror_split)

  records = [dataset[i] for i in range(min(sample_size, len(dataset)))]
  return wtq_records_to_eval(records)


# ------------------------------------------------------------ T2-RAGBench
def t2_records_to_eval(
  records: list[dict],
) -> tuple[list[EvalSample], dict[str, str]]:
  """Convert T2-RAGBench records (HF schema: id, context_id, question,
  program_answer, original_answer, context) into samples + contexts.

  Contexts are full financial-document extracts mixing prose and tables;
  questions are context-independent, so retrieval across all documents is
  a fair test.
  """
  samples: list[EvalSample] = []
  contexts: dict[str, str] = {}

  for record in records:
    context_id = str(record["context_id"])
    contexts.setdefault(context_id, record["context"])
    answers = [str(record["program_answer"])]
    original = str(record.get("original_answer") or "").strip()
    if original and original not in answers:
      answers.append(original)
    samples.append(
      EvalSample(
        id=str(record["id"]),
        question=record["question"],
        answers=answers,
        golden_sources=[context_id],
        dataset="t2-ragbench",
      )
    )
  return samples, contexts


def load_t2ragbench(
  sample_size: int = 50,
  subset: str = "FinQA",
  split: str = "test",
) -> tuple[list[EvalSample], dict[str, str]]:
  """Load a T2-RAGBench subset (needs `pip install tablerag[evals]`).

  Args:
    sample_size: number of QA pairs (their contexts form the corpus).
    subset: "FinQA" | "ConvFinQA" | "TAT-DQA".
    split: dataset split name.
  """
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise ImportError(
      "T2-RAGBench requires the 'datasets' package: "
      "pip install tablerag[evals]"
    ) from exc

  try:
    dataset = load_dataset("G4KMU/t2-ragbench", subset, split=split)
  except ValueError:
    # Config layout may differ; fall back to the default config.
    dataset = load_dataset("G4KMU/t2-ragbench", split=split)

  records = [dataset[i] for i in range(min(sample_size, len(dataset)))]
  return t2_records_to_eval(records)
