"""Deterministic query classifier: no LLM call, no latency, no quota.

"compute" means the answer requires aggregating over table rows (sums,
averages, rates, counts, comparisons across groups). "lookup" means the
answer can be read directly from retrieved context.

The default English pattern list can be extended or trimmed via
RegexClassifier / classify_query(extra_patterns=, disable_patterns=), or
replaced entirely with a custom callable on TableRAGPipeline(classifier=).
"""

from __future__ import annotations

import re
from typing import Callable

DEFAULT_AGGREGATION_PATTERNS = [
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

# Backward-compatible alias.
_AGGREGATION_PATTERNS = DEFAULT_AGGREGATION_PATTERNS

Classifier = Callable[[str], str]


class RegexClassifier:
  """Pattern-based lookup vs compute classifier.

  Args:
    extra_patterns: regexes added on top of the base list (e.g. French
      ``r"\\bmoyenne\\b"``, domain jargon ``r"\\brun[- ]?rate\\b"``).
    disable_patterns: exact base-list entries to remove (must match a
      string in ``base``, e.g. ``r"\\bbelow\\b"``).
    base: pattern list to start from (default DEFAULT_AGGREGATION_PATTERNS).
  """

  def __init__(
    self,
    extra_patterns: list[str] | None = None,
    disable_patterns: list[str] | None = None,
    base: list[str] | None = None,
  ) -> None:
    patterns = list(base if base is not None else DEFAULT_AGGREGATION_PATTERNS)
    if disable_patterns:
      disabled = set(disable_patterns)
      patterns = [p for p in patterns if p not in disabled]
    if extra_patterns:
      patterns.extend(extra_patterns)
    self.patterns = patterns
    self._compiled = [re.compile(p) for p in patterns]

  def __call__(self, query: str) -> str:
    """Return ``\"compute\"`` or ``\"lookup\"``."""
    lowered = query.lower()
    if any(p.search(lowered) for p in self._compiled):
      return "compute"
    return "lookup"


_DEFAULT = RegexClassifier()


def classify_query(
  query: str,
  *,
  extra_patterns: list[str] | None = None,
  disable_patterns: list[str] | None = None,
) -> str:
  """Return ``\"compute\"`` or ``\"lookup\"``.

  With no kwargs, uses the default English aggregation list. Pass
  ``extra_patterns`` / ``disable_patterns`` for a one-off classification
  without building a RegexClassifier yourself.
  """
  if extra_patterns is None and disable_patterns is None:
    return _DEFAULT(query)
  return RegexClassifier(
    extra_patterns=extra_patterns, disable_patterns=disable_patterns
  )(query)
