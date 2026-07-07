"""Benchmark: TRAG baseline (TF-IDF + naive chunking) vs tablerag on document2.

Reuses the baseline answers already captured in data/document2_test_results.json
(from run_doc2_tests.py) and runs the same 10 queries through the tablerag
pipeline. Writes data/document2_tablerag_results.json and prints a comparison.

Usage:
  python benchmarks/run_doc2_comparison.py            # run tablerag queries
  python benchmarks/run_doc2_comparison.py --report   # print table from saved results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_doc2_tests import EXPECTED, load_queries  # noqa: E402

from tablerag.pipeline import TableRAGPipeline, setup_logging  # noqa: E402

CONTEXT = ROOT / "data" / "document2.txt"
BASELINE_RESULTS = ROOT / "data" / "document2_test_results.json"
OUT = ROOT / "data" / "document2_tablerag_results.json"
API_DELAY_SEC = 6


def run_tablerag() -> list[dict]:
  setup_logging("WARNING")
  pipeline = TableRAGPipeline()
  pipeline.ingest(CONTEXT)

  queries = load_queries()
  results: list[dict] = []
  if OUT.exists():
    results = json.loads(OUT.read_text(encoding="utf-8"))
  done = {r["query"] for r in results}

  for i, query in enumerate(queries, start=1):
    if query in done:
      print(f"[{i}/{len(queries)}] skip (done): {query[:55]}...")
      continue
    print(f"[{i}/{len(queries)}] {query[:65]}...")

    for attempt in range(4):
      try:
        result = pipeline.query(query, top_k=3)
        break
      except Exception as exc:
        if "429" in str(exc) and attempt < 3:
          wait = 15 * (attempt + 1)
          print(f"  rate limited, waiting {wait}s...")
          time.sleep(wait)
        else:
          raise

    results.append(
      {
        "query": query,
        "expected": EXPECTED.get(query, "N/A"),
        "tablerag_answer": result.answer,
        "route": result.route,
        "sql": result.sql,
        "sql_result": repr(result.sql_result) if result.sql_result else None,
        "retrieved_sections": [
          f"{r.block.kind}:{r.block.section[:60]}" for r in result.retrieved
        ],
      }
    )
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    time.sleep(API_DELAY_SEC)

  print(f"\nWrote {OUT}")
  return results


def print_report() -> None:
  tablerag_results = (
    json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
  )
  baseline_results = (
    json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    if BASELINE_RESULTS.exists()
    else []
  )
  baseline_by_query = {r["query"]: r for r in baseline_results}

  for i, result in enumerate(tablerag_results, start=1):
    query = result["query"]
    baseline = baseline_by_query.get(query, {})
    print("=" * 100)
    print(f"Q{i}: {query}")
    print(f"  EXPECTED : {result['expected']}")
    print(f"  BASELINE : {baseline.get('vanilla_answer', '(not run)')[:300]}")
    print(f"  TABLERAG : {result['tablerag_answer'][:300]}")
    print(f"  route={result.get('route')}  sql={result.get('sql')}")
    if result.get("sql_result"):
      print(f"  sql_result: {result['sql_result'][:200]}")
    print(f"  retrieved: {result['retrieved_sections']}")
  print("=" * 100)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", action="store_true", help="Print report only.")
  args = parser.parse_args()

  if not args.report:
    run_tablerag()
  print_report()


if __name__ == "__main__":
  main()
