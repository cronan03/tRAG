"""Interactive CLI for table RAG queries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag import get_default_model, run_rag, setup_logging

DEFAULT_CONTEXT = Path(__file__).parent / "data" / "tables_context.txt"
EXIT_COMMANDS = {"exit", "quit", "q"}


def print_result(result: dict, *, show_retrieval: bool) -> None:
  if show_retrieval:
    print("\nRetrieved chunks:")
    for i, (chunk, score) in enumerate(result["retrieved"], start=1):
      preview = chunk.splitlines()[0][:80]
      print(f"  [{i}] score={score:.3f} | {preview}...")

  print("\nAnswer:")
  print(result["answer"])
  print()


def run_query(
  query: str,
  context_path: Path,
  *,
  top_k: int,
  model: str,
  show_retrieval: bool,
  log_level: str,
) -> None:
  setup_logging(log_level)
  result = run_rag(query, context_path, top_k=top_k, model=model)
  print_result(result, show_retrieval=show_retrieval)


def interactive_loop(
  context_path: Path,
  *,
  top_k: int,
  model: str,
  show_retrieval: bool,
  log_level: str,
) -> None:
  setup_logging(log_level)
  print("Table RAG — type your questions (exit / quit / q to leave)\n")

  while True:
    try:
      query = input("Ask> ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      break

    if not query:
      continue
    if query.lower() in EXIT_COMMANDS:
      break

    try:
      run_query(
        query,
        context_path,
        top_k=top_k,
        model=model,
        show_retrieval=show_retrieval,
        log_level=log_level,
      )
    except Exception as exc:
      print(f"Error: {exc}\n", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
  load_dotenv()
  default_model = get_default_model()

  parser = argparse.ArgumentParser(
    prog="trag",
    description="Ask questions about table documentation using RAG + Gemini.",
  )
  parser.add_argument(
    "query",
    nargs="?",
    help="Question to ask (omit to start interactive mode)",
  )
  parser.add_argument(
    "--query",
    dest="query_flag",
    help="Question to ask (alternative to positional argument)",
  )
  parser.add_argument(
    "-c",
    "--context",
    type=Path,
    default=DEFAULT_CONTEXT,
    help=f"Path to context document (default: {DEFAULT_CONTEXT.name})",
  )
  parser.add_argument(
    "-k",
    "--top-k",
    type=int,
    default=3,
    help="Number of context chunks to retrieve (default: 3)",
  )
  parser.add_argument(
    "-m",
    "--model",
    default=default_model,
    help=f"Gemini model name (default: {default_model})",
  )
  parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Show retrieved chunks with similarity scores in output",
  )
  parser.add_argument(
    "--log-level",
    default=os.getenv("LOG_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Pipeline log level (default: INFO, or LOG_LEVEL env var)",
  )
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress pipeline logs (sets log level to WARNING)",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  query = args.query_flag or args.query
  context_path: Path = args.context
  log_level = "WARNING" if args.quiet else args.log_level

  if not context_path.exists():
    print(f"Context file not found: {context_path}", file=sys.stderr)
    return 1

  try:
    if query:
      run_query(
        query,
        context_path,
        top_k=args.top_k,
        model=args.model,
        show_retrieval=args.verbose,
        log_level=log_level,
      )
    else:
      interactive_loop(
        context_path,
        top_k=args.top_k,
        model=args.model,
        show_retrieval=args.verbose,
        log_level=log_level,
      )
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
