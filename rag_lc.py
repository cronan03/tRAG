"""LangChain version of the table RAG pipeline (for side-by-side comparison)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity

from rag import (
  chunk_document,
  chunk_preview,
  get_default_model,
  load_context,
  logger,
  setup_logging,
)

SYSTEM_PROMPT = (
  "You are a helpful assistant that answers questions about database tables "
  "using only the provided context. You may compute simple aggregates "
  "(averages, sums, counts) from sample rows when the data is present. "
  "If the answer cannot be determined from the context, say you don't know."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
  [
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
  ]
)


def get_api_key() -> str:
  load_dotenv()
  api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
  if not api_key:
    raise EnvironmentError(
      "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
    )
  return api_key


def build_retriever(chunks: list[str], top_k: int) -> TFIDFRetriever:
  documents = [
    Document(page_content=chunk, metadata={"chunk_id": index})
    for index, chunk in enumerate(chunks)
  ]
  return TFIDFRetriever.from_documents(documents, k=top_k)


def score_all_chunks(retriever: TFIDFRetriever, query: str) -> list[tuple[str, float]]:
  query_vec = retriever.vectorizer.transform([query])
  scores = cosine_similarity(retriever.tfidf_array, query_vec).reshape(-1)
  ranked = sorted(
    ((doc.page_content, float(score)) for doc, score in zip(retriever.docs, scores)),
    key=lambda item: item[1],
    reverse=True,
  )
  return ranked


def build_chain(model: str | None = None):
  llm = ChatGoogleGenerativeAI(
    model=model or get_default_model(),
    google_api_key=get_api_key(),
    temperature=0,
  )
  return PROMPT_TEMPLATE | llm | StrOutputParser()


def run_rag_lc(
  query: str,
  context_path: Path,
  top_k: int = 3,
  model: str | None = None,
) -> dict:
  """Run the LangChain RAG pipeline (TF-IDF retriever + ChatGoogleGenerativeAI)."""
  resolved_model = model or get_default_model()

  logger.info("=" * 60)
  logger.info("RAG pipeline start (LangChain)")
  logger.info("Query: %r", query)
  logger.info("Context: %s", context_path)
  logger.info("Model: %s | top_k: %d", resolved_model, top_k)

  logger.info("[1/5] Loading context document")
  context = load_context(context_path)
  logger.info("  Loaded %d characters from %s", len(context), context_path.name)

  logger.info("[2/5] Chunking document (same splitter as rag.py)")
  chunks = chunk_document(context)
  logger.info("  Split into %d chunks", len(chunks))
  for index, chunk in enumerate(chunks, start=1):
    title = chunk.strip().splitlines()[0] if chunk.strip() else "(empty)"
    logger.debug("  Chunk %d (%d chars): %s", index, len(chunk), title)

  logger.info("[3/5] Retrieving relevant chunks (LangChain TFIDFRetriever)")
  retriever = build_retriever(chunks, top_k=top_k)
  all_hits = score_all_chunks(retriever, query)
  hits = all_hits[:top_k]
  logger.info("  Similarity scores for all chunks:")
  for index, (chunk, score) in enumerate(all_hits, start=1):
    title = chunk.strip().splitlines()[0] if chunk.strip() else "(empty)"
    marker = " <-- selected" if index <= top_k else ""
    logger.info("    [%d] score=%.4f | %s%s", index, score, title, marker)

  if not hits:
    logger.warning("  No chunks retrieved — context may be empty or query unmatched")

  logger.info("  Top %d chunks passed to the LLM:", len(hits))
  for index, (chunk, score) in enumerate(hits, start=1):
    logger.info("  --- Retrieved chunk %d (score=%.4f) ---", index, score)
    logger.info("%s", chunk_preview(chunk))

  retrieved_chunks = [chunk for chunk, _ in hits]
  context_block = "\n\n---\n\n".join(retrieved_chunks)

  logger.info("[4/5] Building LangChain prompt chain")
  prompt_messages = PROMPT_TEMPLATE.invoke(
    {"context": context_block, "question": query}
  )
  prompt_text = "\n".join(
    f"{message.type.upper()}: {message.content}" for message in prompt_messages.messages
  )
  logger.info("  Prompt length: %d characters", len(prompt_text))
  logger.debug("  Full prompt:\n%s", prompt_text)

  logger.info("[5/5] Generating answer with ChatGoogleGenerativeAI")
  try:
    chain = build_chain(model=resolved_model)
    answer = chain.invoke({"context": context_block, "question": query})
  except Exception:
    logger.exception("LangChain Gemini call failed")
    raise

  if not answer:
    logger.warning("Gemini returned an empty response")
  else:
    logger.info("Gemini response received (%d chars)", len(answer))

  logger.info("  Answer: %r", answer)
  logger.info("RAG pipeline complete (LangChain)")
  logger.info("=" * 60)

  return {
    "query": query,
    "retrieved": hits,
    "all_scores": all_hits,
    "prompt": prompt_text,
    "model": resolved_model,
    "answer": answer,
    "backend": "langchain",
  }


def main_cli(argv: list[str] | None = None) -> int:
  import argparse

  parser = argparse.ArgumentParser(description="Run LangChain RAG for comparison.")
  parser.add_argument("query", help="Question to ask")
  parser.add_argument(
    "-c",
    "--context",
    type=Path,
    default=Path(__file__).parent / "data" / "tables_context.txt",
  )
  parser.add_argument("-k", "--top-k", type=int, default=3)
  parser.add_argument("-m", "--model", default=None)
  parser.add_argument(
    "--log-level",
    default=os.getenv("LOG_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  args = parser.parse_args(argv)

  setup_logging(args.log_level)
  try:
    result = run_rag_lc(args.query, args.context, top_k=args.top_k, model=args.model)
  except Exception as exc:
    print(f"Error: {exc}", file=__import__("sys").stderr)
    return 1

  print("\nAnswer:")
  print(result["answer"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main_cli())
