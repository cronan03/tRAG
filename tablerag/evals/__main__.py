"""CLI for the evaluation harness.

Examples:
  python -m tablerag.evals doc2 --offline                 # free, retrieval-only
  python -m tablerag.evals doc2 --generate --compute      # full pipeline eval
  python -m tablerag.evals wtq --sample-size 50           # WTQ retrieval eval
  python -m tablerag.evals t2 --sample-size 25 --subset FinQA
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tablerag.evals.loaders import load_doc2, load_t2ragbench, load_wtq
from tablerag.evals.runner import Evaluator, format_report
from tablerag.pipeline import TableRAGPipeline, setup_logging


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="python -m tablerag.evals",
    description="Evaluate tablerag retrieval/generation on benchmark datasets.",
  )
  parser.add_argument("dataset", choices=["doc2", "wtq", "t2"])
  parser.add_argument("--sample-size", type=int, default=50,
                      help="Questions to evaluate (wtq/t2 only).")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--generate", action="store_true",
                      help="Also generate answers and score Exact Match (LLM calls).")
  parser.add_argument("--compute", action="store_true",
                      help="Enable the DuckDB SQL route during generation.")
  parser.add_argument("--offline", action="store_true",
                      help="Use the hash embedder (no API; weaker retrieval).")
  parser.add_argument("--delay", type=float, default=6.0,
                      help="Seconds between generation calls (rate limits).")
  parser.add_argument("--subset", default="FinQA",
                      help="t2 only: FinQA | ConvFinQA | TAT-DQA.")
  parser.add_argument("--split", default=None,
                      help="Dataset split (wtq default: validation; t2: test).")
  parser.add_argument("--out", type=Path, default=None,
                      help="Write the full report JSON to this path.")
  parser.add_argument("--log-level", default="WARNING")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  setup_logging(args.log_level)

  if args.dataset == "doc2":
    samples, contexts = load_doc2()
  elif args.dataset == "wtq":
    samples, contexts = load_wtq(
      sample_size=args.sample_size, split=args.split or "validation"
    )
  else:
    samples, contexts = load_t2ragbench(
      sample_size=args.sample_size, subset=args.subset, split=args.split or "test"
    )

  from tablerag.providers import gemini_embedder, gemini_generator

  # Generation (LLM) is only exercised when --generate is set.
  generator = gemini_generator() if args.generate else None
  if args.offline:
    from tablerag.index.embedder import HashEmbedder

    embedder = HashEmbedder()
  else:
    embedder = gemini_embedder()

  pipeline = TableRAGPipeline(
    generator=generator, embedder=embedder, enable_compute=args.compute
  )

  evaluator = Evaluator(
    pipeline,
    top_k=args.top_k,
    generate=args.generate,
    delay_sec=args.delay if args.generate else 0.0,
  )
  report = evaluator.run(samples, contexts, dataset_name=args.dataset)

  print(format_report(report))
  if args.out:
    args.out.write_text(
      json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report written to {args.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
