"""End-to-end example: tablerag retriever inside a standard LangChain chain.

Everything below the TableRetrieverManager lines is vanilla LangChain (LCEL)
that a developer would already have; tablerag only replaces the ingestion
and retrieval layer.

Run:  python examples/langchain_table_rag.py "your question"
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_lc import get_api_key
from tablerag.generate import get_default_model
from tablerag.integrations.langchain import TableRetrieverManager
from tablerag.providers import gemini_embedder

ROOT = Path(__file__).parent.parent
DOC = ROOT / "data" / "document2.txt"


def format_docs(docs) -> str:
  return "\n\n---\n\n".join(doc.page_content for doc in docs)


def main() -> None:
  question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is the corrected net_rev_eur for DE-442 on 2025-06-07?"
  )

  # --- tablerag: table-aware ingest + retrieval ---
  # Bring your own embeddings; here we use Gemini via the provider helper.
  manager = TableRetrieverManager(embedder=gemini_embedder())
  manager.ingest(DOC)
  retriever = manager.as_retriever(k=3)

  # --- standard LangChain from here on ---
  llm = ChatGoogleGenerativeAI(
    model=get_default_model(), google_api_key=get_api_key(), temperature=0
  )
  prompt = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "Answer using only the provided context. Tables are markdown with "
        "section titles and correction notes. If unknown, say you don't know."
        "\n\nContext:\n{context}",
      ),
      ("human", "{question}"),
    ]
  )
  chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )

  print(f"Q: {question}")
  print(f"A: {chain.invoke(question)}")


if __name__ == "__main__":
  main()
