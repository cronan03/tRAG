"""Deterministic query classifier: no LLM call, no latency, no quota.

"compute" means the answer requires aggregating over table rows (sums,
averages, rates, counts, comparisons across groups). "lookup" means the
answer can be read directly from retrieved context.
"""

from __future__ import annotations

import re

_AGGREGATION_PATTERNS = [
  r"\btotal\b",
  r"\bsum\b",
  r"\baverage\b",
  r"\bavg\b",
  r"\bmean of\b",
  r"\bmedian\b",
  r"\bcount\b",
  r"\bhow many\b",
  r"\bnumber of\b",
  r"\brate\b",
  r"\bpercentage\b",
  r"\bpercent\b",
  r"\bratio\b",
  r"\bper\s",
  r"\bacross all\b",
  r"\bcombined\b",
  r"\bstandard deviation\b",
  r"\bvariance\b",
  r"\bhighest\b",
  r"\blowest\b",
  r"\bmaximum\b",
  r"\bminimum\b",
  r"\bmax\b",
  r"\bmin\b",
  r"\bmost\b",
  r"\bleast\b",
  r"\bbelow\b",
  r"\babove\b",
  r"\bmore than\b",
  r"\bless than\b",
]

_COMPILED = [re.compile(p) for p in _AGGREGATION_PATTERNS]


def classify_query(query: str) -> str:
  """Return "compute" or "lookup"."""
  lowered = query.lower()
  if any(p.search(lowered) for p in _COMPILED):
    return "compute"
  return "lookup"
