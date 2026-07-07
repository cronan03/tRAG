"""Automated scoring: answer matching + retrieval metrics.

Answer matching is deliberately pragmatic (this is a benchmark harness, not
a leaderboard submission):

- Text: normalized containment. Gold "CH-902" matches a prediction that
  mentions CH-902 anywhere.
- Numbers: any number found in the prediction is compared to the gold
  number with a relative tolerance (default 1%). Thousands separators,
  currency symbols, and percent signs are stripped; a fraction/percent
  mismatch (0.0353 vs 3.53%) is also accepted.
"""

from __future__ import annotations

import math
import re

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*([kmb])?", re.IGNORECASE)
_STRIP_RE = re.compile(r"[$€£%]")

_SUFFIX_SCALE = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def normalize_text(text: str) -> str:
  return " ".join(_STRIP_RE.sub("", text.lower()).split())


def parse_number(text: str) -> float | None:
  """Parse a lone number from a gold answer ('1,965,043.30' -> 1965043.3)."""
  cleaned = _STRIP_RE.sub("", text.strip().lower()).replace(",", "")
  cleaned = cleaned.removesuffix("k")
  try:
    value = float(cleaned)
  except ValueError:
    return None
  if text.strip().lower().endswith("k"):
    value *= 1000
  return value


def numbers_in(text: str) -> list[float]:
  """All numbers in text, expanding k/m/b suffixes ('$42k' -> 42000)."""
  values = []
  for match in _NUMBER_RE.finditer(text):
    number = float(match.group(0).rstrip("kmbKMB ").replace(",", ""))
    suffix = match.group(1)
    if suffix:
      number *= _SUFFIX_SCALE[suffix.lower()]
    values.append(number)
  return values


def _numbers_close(a: float, b: float, rel_tol: float) -> bool:
  if math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9):
    return True
  # Percent vs fraction (3.53 vs 0.0353) in either direction.
  return math.isclose(a, b * 100, rel_tol=rel_tol) or math.isclose(
    a * 100, b, rel_tol=rel_tol
  )


def _matches_one(prediction: str, gold: str, rel_tol: float) -> bool:
  pred_norm = normalize_text(prediction)
  gold_norm = normalize_text(gold)
  if gold_norm and gold_norm in pred_norm:
    return True

  gold_num = parse_number(gold)
  if gold_num is None:
    return False
  return any(_numbers_close(num, gold_num, rel_tol) for num in numbers_in(prediction))


def answer_match(
  prediction: str,
  gold_answers: list[str],
  *,
  require_all: bool = False,
  rel_tol: float = 0.01,
) -> bool:
  """True when the prediction matches the gold answer(s).

  Args:
    prediction: model output.
    gold_answers: accepted gold strings (any one suffices by default).
    require_all: prediction must match every gold answer (e.g. a question
      whose answer is two numbers).
    rel_tol: relative tolerance for numeric comparison.
  """
  if not gold_answers:
    return False
  checks = (_matches_one(prediction, gold, rel_tol) for gold in gold_answers)
  return all(checks) if require_all else any(checks)


# ------------------------------------------------------- retrieval metrics
def recall_at_k(hit_ranks: list[int | None], k: int) -> float:
  """Fraction of samples whose first golden block ranked within top-k."""
  if not hit_ranks:
    return 0.0
  hits = sum(1 for rank in hit_ranks if rank is not None and rank <= k)
  return hits / len(hit_ranks)


def mrr_at_k(hit_ranks: list[int | None], k: int) -> float:
  """Mean reciprocal rank, counting ranks beyond k as 0."""
  if not hit_ranks:
    return 0.0
  total = sum(
    1.0 / rank for rank in hit_ranks if rank is not None and rank <= k
  )
  return total / len(hit_ranks)
