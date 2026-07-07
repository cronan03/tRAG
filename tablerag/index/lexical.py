"""IDF-weighted lexical overlap scorer.

Complements embeddings: table queries are full of exact identifiers
(column names like `net_rev_usd`, SKUs like WB-8841, dates) that vector
similarity blurs but exact token match nails.
"""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")
_SUBTOKEN_SPLIT_RE = re.compile(r"[_\-\.]")


def tokenize(text: str) -> list[str]:
  """Emit compound identifiers plus their parts.

  `net_rev_usd` -> [net_rev_usd, net, rev, usd] so a query phrased as
  "net revenue USD" still overlaps the column name, and "NA" matches
  region codes like NA-EAST.
  """
  tokens: list[str] = []
  for token in _TOKEN_RE.findall(text.lower()):
    tokens.append(token)
    parts = [p for p in _SUBTOKEN_SPLIT_RE.split(token) if p]
    if len(parts) > 1:
      tokens.extend(parts)
  return tokens


class LexicalScorer:
  def __init__(self) -> None:
    self._doc_tokens: list[set[str]] = []
    self._idf: dict[str, float] = {}

  def add_documents(self, texts: list[str]) -> None:
    for text in texts:
      self._doc_tokens.append(set(tokenize(text)))
    self._recompute_idf()

  def _recompute_idf(self) -> None:
    n = len(self._doc_tokens)
    counts: dict[str, int] = {}
    for tokens in self._doc_tokens:
      for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    self._idf = {
      token: math.log((n + 1) / (count + 0.5)) for token, count in counts.items()
    }

  def _match_weight(self, query_token: str, doc_tokens: set[str]) -> float:
    if query_token in doc_tokens:
      return self._idf.get(query_token, 0.0)
    # Prefix match handles abbreviation drift between natural language and
    # column names: "revenue" ~ "rev", "stores" ~ "store".
    if len(query_token) >= 3:
      best = 0.0
      for doc_token in doc_tokens:
        if len(doc_token) >= 3 and (
          doc_token.startswith(query_token) or query_token.startswith(doc_token)
        ):
          best = max(best, 0.8 * self._idf.get(doc_token, 0.0))
      return best
    return 0.0

  def score(self, query: str, doc_index: int) -> float:
    """Normalized IDF-weighted overlap between query and document tokens."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
      return 0.0
    doc_tokens = self._doc_tokens[doc_index]
    matched = sum(self._match_weight(t, doc_tokens) for t in query_tokens)
    total = sum(self._idf.get(t, self._max_idf()) for t in query_tokens)
    return matched / total if total > 0 else 0.0

  def _max_idf(self) -> float:
    return max(self._idf.values()) if self._idf else 1.0

  def __len__(self) -> int:
    return len(self._doc_tokens)
