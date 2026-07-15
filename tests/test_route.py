"""Configurable query routing (T1-2)."""

from __future__ import annotations

import pytest

from tablerag import TableRAGPipeline
from tablerag.generate import CallableGenerator
from tablerag.index.embedder import HashEmbedder
from tablerag.route import RegexClassifier, classify_query

DOC = """=== SALES ===
Weekly revenue by region.

| region | net_rev_usd |
| --- | --- |
| NA-EAST | 640220.10 |
| NA-WEST | 325044.80 |
"""


def test_default_classify_query_unchanged():
  assert classify_query("What is the total revenue?") == "compute"
  assert classify_query("What does the oh column mean?") == "lookup"


def test_extra_patterns_enable_french_keyword():
  assert classify_query("Quelle est la moyenne de oh?") == "lookup"
  assert (
    classify_query(
      "Quelle est la moyenne de oh?", extra_patterns=[r"\bmoyenne\b"]
    )
    == "compute"
  )


def test_disable_patterns_removes_base_trigger():
  assert classify_query("Which variants are below the threshold?") == "compute"
  assert (
    classify_query(
      "Which variants are below the threshold?",
      disable_patterns=[r"\bbelow\b"],
    )
    == "lookup"
  )


def test_regex_classifier_combines_extra_and_disable():
  clf = RegexClassifier(
    extra_patterns=[r"\brun[- ]?rate\b"],
    disable_patterns=[r"\bbelow\b"],
  )
  assert clf("What is our run-rate?") == "compute"
  assert clf("items below threshold") == "lookup"
  assert clf("total revenue") == "compute"


def test_pipeline_extra_compute_patterns():
  pipe = TableRAGPipeline(
    generator=CallableGenerator(lambda p: "ok"),
    embedder=HashEmbedder(),
    extra_compute_patterns=[r"\bmoyenne\b"],
  )
  pipe.ingest(DOC)
  result = pipe.query("Quelle est la moyenne?")
  # SQL may succeed or fall back; entry route should have been compute
  # (hybrid if SQL worked, lookup only if SQL declined — either way we
  # verify classifier via a direct call).
  assert pipe.classifier("Quelle est la moyenne?") == "compute"
  assert result.route in ("hybrid", "lookup")


def test_pipeline_custom_classifier():
  pipe = TableRAGPipeline(
    generator=CallableGenerator(lambda p: "ok"),
    embedder=HashEmbedder(),
    classifier=lambda q: "compute" if "math" in q.lower() else "lookup",
  )
  pipe.ingest(DOC)
  assert pipe.classifier("do the math please") == "compute"
  assert pipe.classifier("total revenue") == "lookup"  # ignores default list


def test_query_route_force_lookup_skips_sql():
  prompts = []

  def llm(prompt: str) -> str:
    prompts.append(prompt)
    if prompt.rstrip().endswith("SQL:"):
      return "SELECT 1 AS x"
    return "answer"

  pipe = TableRAGPipeline(
    generator=CallableGenerator(llm),
    embedder=HashEmbedder(),
  )
  pipe.ingest(DOC)
  result = pipe.query("What is the total net revenue?", route="lookup")
  assert result.route == "lookup"
  assert result.sql is None
  assert not any(p.rstrip().endswith("SQL:") for p in prompts)


def test_query_route_force_compute():
  prompts = []

  def llm(prompt: str) -> str:
    prompts.append(prompt)
    if prompt.rstrip().endswith("SQL:"):
      return "SELECT 99 AS x"
    return "answer"

  pipe = TableRAGPipeline(
    generator=CallableGenerator(llm),
    embedder=HashEmbedder(),
  )
  pipe.ingest(DOC)
  # No aggregation keyword — would be lookup under default classifier.
  result = pipe.query("What is NA-WEST net_rev_usd?", route="compute")
  assert result.route == "hybrid"
  assert result.sql_result == [{"x": 99}]


def test_query_invalid_route_raises():
  pipe = TableRAGPipeline(
    generator=CallableGenerator(lambda p: "ok"),
    embedder=HashEmbedder(),
    enable_compute=False,
  )
  with pytest.raises(ValueError, match="route must be"):
    pipe.query("anything", route="hybrid")


def test_classifier_wins_over_extra_patterns():
  pipe = TableRAGPipeline(
    embedder=HashEmbedder(),
    enable_compute=False,
    classifier=lambda q: "lookup",
    extra_compute_patterns=[r"\btotal\b"],
  )
  assert pipe.classifier("total revenue") == "lookup"
