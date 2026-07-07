"""Simple RAG pipeline for table documentation (vanilla Python retrieval + Gemini)."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger("trag")


def setup_logging(level: str = "INFO") -> None:
  numeric_level = getattr(logging, level.upper(), logging.INFO)
  root = logging.getLogger()
  if not root.handlers:
    logging.basicConfig(
      level=numeric_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      datefmt="%H:%M:%S",
    )
  else:
    root.setLevel(numeric_level)
  logger.setLevel(numeric_level)


def chunk_preview(chunk: str, max_lines: int = 3) -> str:
  lines = chunk.strip().splitlines()[:max_lines]
  return "\n".join(f"    {line}" for line in lines)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def chunk_document(text: str, separator: str = "\n---\n") -> list[str]:
    """Split context into sections; fall back to paragraph chunks."""
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) > 1:
        return parts

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def build_tfidf_index(chunks: list[str]) -> tuple[list[Counter[str]], dict[str, float]]:
  """Return per-chunk term frequencies and inverse document frequencies."""
  doc_freq: Counter[str] = Counter()
  chunk_counters: list[Counter[str]] = []

  for chunk in chunks:
    tokens = tokenize(chunk)
    counter = Counter(tokens)
    chunk_counters.append(counter)
    for term in counter:
      doc_freq[term] += 1

  num_docs = len(chunks)
  idf = {
    term: math.log((1 + num_docs) / (1 + freq)) + 1.0
    for term, freq in doc_freq.items()
  }
  return chunk_counters, idf


def tfidf_vector(counter: Counter[str], idf: dict[str, float]) -> dict[str, float]:
  total = sum(counter.values()) or 1
  return {term: (count / total) * idf.get(term, 0.0) for term, count in counter.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
  common = set(vec_a) & set(vec_b)
  dot = sum(vec_a[t] * vec_b[t] for t in common)
  norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
  norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
  if norm_a == 0 or norm_b == 0:
    return 0.0
  return dot / (norm_a * norm_b)


def retrieve(
  query: str,
  chunks: list[str],
  top_k: int = 3,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
  counters, idf = build_tfidf_index(chunks)
  query_tokens = tokenize(query)
  query_vec = tfidf_vector(Counter(query_tokens), idf)

  logger.debug("Query tokens: %s", query_tokens)

  scored = [
    (chunk, cosine_similarity(query_vec, tfidf_vector(counter, idf)))
    for chunk, counter in zip(chunks, counters)
  ]
  scored.sort(key=lambda item: item[1], reverse=True)
  return scored[:top_k], scored


def load_context(path: Path) -> str:
  if not path.exists():
    raise FileNotFoundError(f"Context file not found: {path}")
  return path.read_text(encoding="utf-8")


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
  context_block = "\n\n---\n\n".join(retrieved_chunks)
  return (
    "You are a helpful assistant that answers questions about database tables "
    "using only the provided context. You may compute simple aggregates "
    "(averages, sums, counts) from sample rows when the data is present. "
    "If the answer cannot be determined from the context, say you don't know.\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {query}\n\n"
    "Answer:"
  )


def get_default_model() -> str:
  load_dotenv()
  return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def ask_gemini(prompt: str, model: str | None = None) -> str:
  load_dotenv()
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise EnvironmentError(
      "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

  resolved_model = model or get_default_model()
  logger.info("Calling Gemini model=%s (prompt length=%d chars)", resolved_model, len(prompt))

  client = genai.Client(api_key=api_key)
  try:
    response = client.models.generate_content(
      model=resolved_model,
      contents=prompt,
    )
  except Exception:
    logger.exception("Gemini API call failed")
    raise

  answer = response.text or ""
  if not answer:
    logger.warning("Gemini returned an empty response")
  else:
    logger.info("Gemini response received (%d chars)", len(answer))

  return answer


def run_rag(
  query: str,
  context_path: Path,
  top_k: int = 3,
  model: str | None = None,
) -> dict:
  resolved_model = model or get_default_model()

  logger.info("=" * 60)
  logger.info("RAG pipeline start")
  logger.info("Query: %r", query)
  logger.info("Context: %s", context_path)
  logger.info("Model: %s | top_k: %d", resolved_model, top_k)

  logger.info("[1/5] Loading context document")
  context = load_context(context_path)
  logger.info("  Loaded %d characters from %s", len(context), context_path.name)

  logger.info("[2/5] Chunking document")
  chunks = chunk_document(context)
  logger.info("  Split into %d chunks", len(chunks))
  for index, chunk in enumerate(chunks, start=1):
    title = chunk.strip().splitlines()[0] if chunk.strip() else "(empty)"
    logger.debug("  Chunk %d (%d chars): %s", index, len(chunk), title)

  logger.info("[3/5] Retrieving relevant chunks (TF-IDF)")
  hits, all_hits = retrieve(query, chunks, top_k=top_k)
  logger.info("  Similarity scores for all chunks:")
  for index, (chunk, score) in enumerate(all_hits, start=1):
    title = chunk.strip().splitlines()[0] if chunk.strip() else "(empty)"
    marker = " <-- selected" if (chunk, score) in hits else ""
    logger.info("    [%d] score=%.4f | %s%s", index, score, title, marker)

  if not hits:
    logger.warning("  No chunks retrieved — context may be empty or query unmatched")

  logger.info("  Top %d chunks passed to the LLM:", len(hits))
  for index, (chunk, score) in enumerate(hits, start=1):
    logger.info("  --- Retrieved chunk %d (score=%.4f) ---", index, score)
    logger.info("%s", chunk_preview(chunk))

  retrieved_chunks = [chunk for chunk, _ in hits]

  logger.info("[4/5] Building prompt")
  prompt = build_prompt(query, retrieved_chunks)
  logger.info("  Prompt length: %d characters", len(prompt))
  logger.debug("  Full prompt:\n%s", prompt)

  logger.info("[5/5] Generating answer with Gemini")
  answer = ask_gemini(prompt, model=resolved_model)
  logger.info("  Answer: %r", answer)
  logger.info("RAG pipeline complete")
  logger.info("=" * 60)

  return {
    "query": query,
    "retrieved": hits,
    "all_scores": all_hits,
    "prompt": prompt,
    "model": resolved_model,
    "answer": answer,
  }
