"""Compare vanilla vs LangChain RAG on document2 test queries."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag import run_rag, setup_logging
from rag_lc import run_rag_lc

CONTEXT = Path(__file__).parent / "data" / "document2.txt"
QUERIES_FILE = Path(__file__).parent / "data" / "document2_test_queries.txt"
OUT = Path(__file__).parent / "data" / "document2_test_results.json"
API_DELAY_SEC = 14

EXPECTED = {
  "What is the corrected net_rev_eur for DE-442 on 2025-06-07?": (
    "391005.15 EUR (finance restatement 6/20; was 388220.15)"
  ),
  "What is the average stock_quantity for products in the warehouse?": (
    "No stock_quantity column in warehouse snapshot; if using on-hand (oh): "
    "851.86 avg across 7 SKUs; if avail: 685.43. Question uses wrong field name."
  ),
  "How many units of WB-8841 are available to sell according to Shopify vs WMS?": (
    "WMS avail: 3310; Shopify sellable: 3295 (reconciliation notes 2025-06-27)"
  ),
  "What is the refund rate for NA-WEST stores for the week ending 2025-06-14?": (
    "208 refund_cnt / 5890 units_rev ≈ 3.53% (S0201: 189/4988 + S0333: 19/902 combined)"
  ),
  "What is the total net revenue in USD across all NA stores for 2025-06-14?": (
    "$1,965,043.30 (640220.10 + 325044.80 + 862924.40 + 136854.00)"
  ),
  "How many support tickets mention firmware v3.2.1 and requested a refund?": (
    "0 from listed tickets (44821 mentions v3.2.1, refund=no). "
    "Cluster note says 37 mention v3.2.1 but refund status not given for all."
  ),
  "What is the attributable incremental revenue from campaign SummerGlow?": (
    "$42,000 per BI last-touch for week ending 6/14; brand team claims $68,000 blended. "
    "Not in store rollups."
  ),
  "What is the average basket size in NA?": (
    "Weighted from store rollups for 2025-06-14: ~$156.60 simple avg of avg_basket "
    "(159.62, 142.07, 173.00, 151.72). Ignore email ~$160 (non-authoritative)."
  ),
  "Which product variants are below the reorder threshold?": (
    "Per reconciliation: CH-902 family (CH-902 + CH-902-L) below combined OH threshold of 100. "
    "Also LP-35-B oh=18 individually low."
  ),
  "What is the APAC refund count for SG-05 on 2025-06-07?": (
    "Unknown — export truncated; SG-05 refunds field is empty/missing"
  ),
}


def load_queries() -> list[str]:
  lines = QUERIES_FILE.read_text(encoding="utf-8").splitlines()
  return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def run_with_retry(fn, query: str, retries: int = 4):
  for attempt in range(retries):
    try:
      return fn(query, CONTEXT, top_k=3)
    except Exception as exc:
      if "429" in str(exc) and attempt < retries - 1:
        wait = API_DELAY_SEC * (attempt + 1)
        print(f"  rate limited, waiting {wait}s...")
        time.sleep(wait)
      else:
        raise


def main() -> None:
  setup_logging("WARNING")
  queries = load_queries()
  results = []
  if OUT.exists():
    results = json.loads(OUT.read_text(encoding="utf-8"))
  done = {r["query"] for r in results}

  for i, query in enumerate(queries, start=1):
    if query in done:
      print(f"[{i}/{len(queries)}] skip (done): {query[:50]}...")
      continue

    print(f"[{i}/{len(queries)}] {query[:60]}...")
    vanilla = run_with_retry(run_rag, query)
    time.sleep(API_DELAY_SEC)
    lc = run_with_retry(run_rag_lc, query)
    time.sleep(API_DELAY_SEC)

    results.append(
      {
        "query": query,
        "expected": EXPECTED.get(query, "N/A"),
        "vanilla_answer": vanilla["answer"],
        "lc_answer": lc["answer"],
        "vanilla_top_chunk": vanilla["retrieved"][0][0].splitlines()[0][:80] if vanilla["retrieved"] else "",
        "lc_top_chunk": lc["retrieved"][0][0].splitlines()[0][:80] if lc["retrieved"] else "",
      }
    )
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

  print(f"\nWrote {OUT}")


if __name__ == "__main__":
  main()
