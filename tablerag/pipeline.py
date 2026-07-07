"""High-level tablerag pipeline: ingest heterogeneous docs, answer queries."""

from __future__ import annotations

import logging
from pathlib import Path

from tablerag.chunk import split_table_block, split_text_block
from tablerag.compute.sandbox import TableSandbox
from tablerag.compute.sql_agent import SQLAgent
from tablerag.generate import GeminiGenerator, get_default_model
from tablerag.index.backends import LangChainVectorStoreBackend, VectorBackend
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import Embedder, GeminiEmbedder
from tablerag.models import Block, QueryResult, TableBlock, TextBlock
from tablerag.parse import parse_document
from tablerag.route import classify_query

logger = logging.getLogger("tablerag")

SYSTEM_PROMPT = (
  "You are a data analyst answering questions about business documents that "
  "mix prose and tables. Use only the provided context. Tables are given in "
  "markdown with their section titles and any correction notes. You may "
  "compute aggregates (sums, averages, rates) from table rows when the data "
  "is present. Prefer authoritative table data over informal narrative "
  "(emails, hearsay). If a value is missing or the answer cannot be "
  "determined from the context, say you don't know."
)


def setup_logging(level: str = "INFO") -> None:
  numeric = getattr(logging, level.upper(), logging.INFO)
  root = logging.getLogger()
  if not root.handlers:
    logging.basicConfig(
      level=numeric,
      format="%(asctime)s [%(levelname)s] %(message)s",
      datefmt="%H:%M:%S",
    )
  else:
    root.setLevel(numeric)
  logger.setLevel(numeric)


def render_block(block: Block) -> str:
  """Render a block for the LLM context window."""
  if isinstance(block, TableBlock):
    parts = [f"### {block.section}", block.markdown]
    if block.context_notes:
      parts.append(f"Notes: {block.context_notes}")
    return "\n".join(parts)
  return f"### {block.section}\n{block.content}"


class TableRAGPipeline:
  def __init__(
    self,
    embedder: Embedder | None = None,
    model: str | None = None,
    max_table_rows: int = 50,
    enable_compute: bool = True,
    embed_model: str | None = None,
    similarity: str = "cosine",
    lexical_weight: float = 0.5,
    max_text_words: int = 300,
    text_overlap_words: int = 0,
    max_table_words: int | None = None,
    table_overlap_rows: int = 0,
    vector_backend: VectorBackend | None = None,
    vectorstore=None,
  ) -> None:
    """
    Args:
      embedder: custom Embedder. Defaults to GeminiEmbedder(embed_model).
      model: Gemini generation model (default: GEMINI_MODEL env var).
      max_table_rows: row cap per table slice (header-aware split).
      enable_compute: enable the DuckDB sandbox + SQL routing.
      embed_model: Gemini embedding model name (ignored when embedder or
        an external backend is provided).
      similarity: "cosine" | "dot" | "euclidean" (in-memory backend only).
      lexical_weight: 0..1 blend of exact-token overlap vs semantic score
        at retrieval time. 0 disables lexical scoring.
      max_text_words: word budget per prose chunk.
      text_overlap_words: words repeated between consecutive prose chunks.
      max_table_words: optional word budget per table slice (derives a row
        cap from average row width; stricter of this and max_table_rows).
      table_overlap_rows: rows repeated between consecutive table slices.
      vector_backend: custom VectorBackend (overrides the in-memory one).
      vectorstore: any LangChain-compatible VectorStore; shorthand for
        vector_backend=LangChainVectorStoreBackend(vectorstore).
    """
    if vectorstore is not None and vector_backend is None:
      vector_backend = LangChainVectorStoreBackend(vectorstore)

    if vector_backend is None and embedder is None:
      embedder = GeminiEmbedder(model=embed_model)

    self.embedder = embedder
    self.index = DualVectorIndex(
      embedder,
      lexical_weight=lexical_weight,
      backend=vector_backend,
      similarity=similarity,
    )
    self.model = model or get_default_model()
    self.max_table_rows = max_table_rows
    self.max_table_words = max_table_words
    self.table_overlap_rows = table_overlap_rows
    self.max_text_words = max_text_words
    self.text_overlap_words = text_overlap_words
    self.enable_compute = enable_compute
    self.sandbox = TableSandbox() if enable_compute else None
    self._generator: GeminiGenerator | None = None
    self._sql_agent: SQLAgent | None = None

  @classmethod
  def from_env(cls) -> TableRAGPipeline:
    return cls()

  @property
  def generator(self) -> GeminiGenerator:
    if self._generator is None:
      self._generator = GeminiGenerator(model=self.model)
    return self._generator

  @property
  def sql_agent(self) -> SQLAgent | None:
    if self.sandbox is None:
      return None
    if self._sql_agent is None:
      self._sql_agent = SQLAgent(self.sandbox, self.generator)
    return self._sql_agent

  # ------------------------------------------------------------- ingest
  def parse(self, path_or_text: str | Path) -> list[Block]:
    """Stage 1: parse a file path or raw text into Text/Table blocks.

    No chunking, no embedding, no API calls.
    """
    path = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if isinstance(path_or_text, Path) or (
      len(str(path_or_text)) < 4096 and path.exists()
    ):
      text = path.read_text(encoding="utf-8")
      source = path.name
    else:
      text = str(path_or_text)
      source = "(inline)"

    logger.info("[parse] %s (%d chars)", source, len(text))
    blocks = parse_document(text, source=source)
    tables = sum(1 for b in blocks if isinstance(b, TableBlock))
    logger.info(
      "[parse] %d blocks (%d tables, %d text)",
      len(blocks),
      tables,
      len(blocks) - tables,
    )
    return blocks

  def chunk(self, blocks: list[Block]) -> list[Block]:
    """Stage 2: apply word/row budgets. Deterministic, no API calls."""
    chunked: list[Block] = []
    for block in blocks:
      if isinstance(block, TableBlock):
        chunked.extend(
          split_table_block(
            block,
            max_rows=self.max_table_rows,
            overlap_rows=self.table_overlap_rows,
            max_words=self.max_table_words,
          )
        )
      elif isinstance(block, TextBlock):
        chunked.extend(
          split_text_block(
            block,
            max_words=self.max_text_words,
            overlap_words=self.text_overlap_words,
          )
        )
      else:
        chunked.append(block)
    logger.info("[chunk] %d blocks -> %d chunks", len(blocks), len(chunked))
    return chunked

  def index_blocks(
    self,
    blocks: list[Block],
    *,
    sandbox_tables: list[TableBlock] | None = None,
  ) -> None:
    """Stage 3: summarize + embed blocks into the vector index (this is
    the only ingest stage that calls the embedding API), and load tables
    into the SQL sandbox.

    Args:
      blocks: (chunked) blocks to index for retrieval.
      sandbox_tables: tables to load into DuckDB. Defaults to the
        TableBlocks found in `blocks`; pass the ORIGINAL unchunked tables
        when you chunked with a small row budget, so SQL always sees
        complete tables.
    """
    logger.info("[index] Embedding + indexing %d blocks (dual-vector)", len(blocks))
    self.index.add_blocks(blocks)

    if self.sandbox is not None:
      tables = (
        sandbox_tables
        if sandbox_tables is not None
        else [b for b in blocks if isinstance(b, TableBlock)]
      )
      if tables:
        names = self.sandbox.load_tables(tables)
        logger.info("[index] Sandbox tables: %s", ", ".join(names))

  def ingest(self, path_or_text: str | Path) -> list[Block]:
    """Convenience: parse -> chunk -> index in one call."""
    blocks = self.parse(path_or_text)
    original_tables = [b for b in blocks if isinstance(b, TableBlock)]
    chunked = self.chunk(blocks)
    self.index_blocks(chunked, sandbox_tables=original_tables)
    return chunked

  # -------------------------------------------------------------- query
  def build_prompt(
    self,
    query: str,
    retrieved: list,
    sql: str | None = None,
    sql_result: list[dict] | None = None,
  ) -> str:
    context = "\n\n---\n\n".join(render_block(r.block) for r in retrieved)
    sql_section = ""
    if sql and sql_result is not None:
      sql_section = (
        "\n\nComputed result (SQL was executed against the full parsed "
        f"tables, so it is exact):\nSQL: {sql}\nResult: {sql_result}\n"
        "Trust this computed result over manual arithmetic."
      )
    return (
      f"{SYSTEM_PROMPT}\n\nContext:\n{context}{sql_section}\n\n"
      f"Question: {query}\n\nAnswer:"
    )

  def query(self, question: str, top_k: int = 3) -> QueryResult:
    logger.info("=" * 60)
    logger.info("tablerag query: %r", question)

    route = "lookup"
    if self.sandbox is not None and len(self.sandbox) > 0:
      route = classify_query(question)
    logger.info("[1/4] Route: %s", route)

    logger.info("[2/4] Retrieving top-%d blocks (dual-vector)", top_k)
    retrieved = self.index.search(question, top_k=top_k)
    for i, r in enumerate(retrieved, start=1):
      logger.info(
        "  [%d] score=%.4f | %s | %s", i, r.score, r.block.kind, r.block.section
      )

    sql, sql_result = None, None
    if route == "compute":
      logger.info("[3/4] SQL agent over %d sandbox tables", len(self.sandbox))
      sql, sql_result = self.sql_agent.answer(question)
      if sql is None:
        logger.info("  SQL not applicable; falling back to lookup context")
        route = "lookup"
      else:
        route = "hybrid"
        logger.info("  SQL result: %s", sql_result)

    prompt = self.build_prompt(question, retrieved, sql, sql_result)
    logger.info("[4/4] Generating with %s (prompt %d chars)", self.model, len(prompt))
    answer = self.generator.generate(prompt)
    logger.info("  Answer: %r", answer[:200])
    logger.info("=" * 60)

    return QueryResult(
      query=question,
      answer=answer,
      route=route,
      retrieved=retrieved,
      sql=sql,
      sql_result=sql_result,
      prompt=prompt,
      model=self.model,
    )
