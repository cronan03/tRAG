"""Interactive CLI for the tablerag pipeline (registered as `trag-tab`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tablerag.pipeline import TableRAGPipeline, setup_logging

DEFAULT_CONTEXT = Path(__file__).parent.parent / "data" / "document2.txt"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="trag-tab",
    description="Table-aware RAG over heterogeneous documents (tablerag).",
  )
  parser.add_argument("--query", "-q", help="Run one query and exit.")
  parser.add_argument(
    "--context",
    "-c",
    default=str(DEFAULT_CONTEXT),
    help=f"Context document path (default: {DEFAULT_CONTEXT.name})",
  )
  parser.add_argument("--top-k", "-k", type=int, default=3, help="Blocks to retrieve.")
  parser.add_argument("--model", "-m", help="Gemini model override.")
  parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging verbosity.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  setup_logging(args.log_level)

  from tablerag.providers import gemini_embedder, gemini_generator

  pipeline = TableRAGPipeline(
    generator=gemini_generator(model=args.model),
    embedder=gemini_embedder(),
  )
  pipeline.ingest(Path(args.context))

  if args.query:
    result = pipeline.query(args.query, top_k=args.top_k)
    print(f"\n{result.answer}\n")
    return 0

  print("tablerag interactive mode. Type a question, or 'exit' to quit.")
  while True:
    try:
      question = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
      break
    if not question or question.lower() in {"exit", "quit"}:
      break
    result = pipeline.query(question, top_k=args.top_k)
    print(f"\n{result.answer}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
