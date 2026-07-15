"""High-level tablerag pipeline: ingest heterogeneous docs, answer queries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from tablerag.chunk import split_table_block, split_text_block
from tablerag.compute.sandbox import TableSandbox
from tablerag.compute.sql_agent import SQLAgent
from tablerag.generate import Generator
from tablerag.index.backends import LangChainVectorStoreBackend, VectorBackend
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import Embedder
from tablerag.models import Block, QueryResult, TableBlock, TextBlock
from tablerag.parse import parse_document
from tablerag.route import classify_query

logger = logging.getLogger("tablerag")

# User-owned persona/domain guidance. Override via TableRAGPipeline(
# system_prompt=...) or per-query with query(..., system_prompt=...).
DEFAULT_SYSTEM_PROMPT = (
  "You are a data analyst answering questions about business documents that "
  "mix prose and tables. Use only the provided context. Prefer authoritative "
  "table data over informal narrative (emails, hearsay). If a value is "
  "missing or the answer cannot be determined from the context, say you "
  "don't know."
)

# Backward-compatible alias (the old name for the module constant).
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# tablerag-owned structural contract: describes how the pipeline formats the
# context (canonical markdown tables, section titles, correction notes).
# Always injected regardless of the user's system_prompt so a custom persona
# cannot accidentally break table reading. prompt_builder= is the explicit
# escape hatch for users who want to own the entire layout.
FORMAT_CONTRACT = (
  "Tables are given in markdown with their section titles and any correction "
  "notes. You may compute aggregates (sums, averages, rates) from table rows "
  "when the data is present."
)

# Signature: (system, context, question, sql=None, sql_result=None) -> prompt
PromptBuilder = Callable[..., str]


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
    generator: Generator | None = None,
    embedder: Embedder | None = None,
    max_table_rows: int = 50,
    enable_compute: bool = True,
    similarity: str = "cosine",
    lexical_weight: float = 0.5,
    max_text_words: int = 300,
    text_overlap_words: int = 0,
    max_table_words: int | None = None,
    table_overlap_rows: int = 0,
    vector_backend: VectorBackend | None = None,
    vectorstore=None,
    system_prompt: str | None = None,
    prompt_builder: PromptBuilder | None = None,
    sql_instructions: str | None = None,
    sql_examples: list[tuple[str, str]] | None = None,
    sql_prompt_template: str | None = None,
    sql_max_retries: int = 1,
  ) -> None:
    """
    tablerag is model-agnostic: bring your own generator and embedder.
    See tablerag.providers for one-line constructors, or
    TableRAGPipeline.from_env() for a Gemini setup from .env.

    Args:
      generator: any Generator (generate(prompt) -> str). Required only for
        query()/generation; retrieval-only usage may omit it.
      embedder: any Embedder (embed(texts) -> vectors). Optional at
        construction; parse/chunk work without it. The requirement is enforced
        lazily the first time embedding is needed (index_blocks/search) unless
        a vector_backend or vectorstore is supplied (those embed themselves).
      max_table_rows: row cap per table slice (header-aware split).
      enable_compute: enable the DuckDB sandbox + SQL routing.
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
      system_prompt: persona/domain instructions for answer generation
        (tone, guardrails, language). Defaults to DEFAULT_SYSTEM_PROMPT.
        tablerag's format contract (how tables appear in the context) is
        always appended, so this cannot break table reading. Can also be
        overridden per query: query(..., system_prompt=...).
      prompt_builder: full-control escape hatch for prompt layout. A callable
        (system, context, question, sql=None, sql_result=None) -> str.
        When set, it replaces the default layout entirely (including the
        format contract and SQL-trust note) — you own the whole prompt.
      sql_instructions: use-case domain rules APPENDED to the SQL agent's
        prompt (unit conventions, fiscal calendars, business definitions,
        tenant filters). The base safety rules stay intact.
      sql_examples: few-shot (question, sql) pairs rendered into the SQL
        prompt — the biggest accuracy lever on quirky schemas.
      sql_prompt_template: full SQL prompt replacement with {schema},
        {question}, {instructions} slots. You own the "output only SQL /
        NO_SQL" contract when overriding.
      sql_max_retries: corrected attempts after a failed SQL execution
        (error fed back to the LLM). Default 1; 0 disables retrying.
    """
    if vectorstore is not None and vector_backend is None:
      vector_backend = LangChainVectorStoreBackend(vectorstore)

    self.embedder = embedder
    self.index = DualVectorIndex(
      embedder,
      lexical_weight=lexical_weight,
      backend=vector_backend,
      similarity=similarity,
    )
    self.model = getattr(generator, "model", "") if generator else ""
    self.max_table_rows = max_table_rows
    self.max_table_words = max_table_words
    self.table_overlap_rows = table_overlap_rows
    self.max_text_words = max_text_words
    self.text_overlap_words = text_overlap_words
    self.enable_compute = enable_compute
    self.sandbox = TableSandbox() if enable_compute else None
    self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    self.prompt_builder = prompt_builder
    self.sql_instructions = sql_instructions
    self.sql_examples = sql_examples
    self.sql_prompt_template = sql_prompt_template
    self.sql_max_retries = sql_max_retries
    self._generator: Generator | None = generator
    self._sql_agent: SQLAgent | None = None

  @classmethod
  def from_env(cls, **kwargs) -> TableRAGPipeline:
    """Convenience: a Gemini pipeline built from .env (GEMINI_API_KEY, etc.).

    Extra kwargs pass through to __init__ (chunking, similarity, ...).
    """
    from tablerag.providers import gemini_embedder, gemini_generator

    return cls(
      generator=gemini_generator(), embedder=gemini_embedder(), **kwargs
    )

  @property
  def generator(self) -> Generator:
    if self._generator is None:
      raise ValueError(
        "No generator configured. Pass generator= to enable answering, e.g. "
        "tablerag.providers.gemini_generator(), or "
        "tablerag.providers.langchain_generator(ChatOpenAI(model='gpt-4o'))."
      )
    return self._generator

  @property
  def sql_agent(self) -> SQLAgent | None:
    if self.sandbox is None:
      return None
    if self._sql_agent is None:
      self._sql_agent = SQLAgent(
        self.sandbox,
        self.generator,
        instructions=self.sql_instructions,
        examples=self.sql_examples,
        prompt_template=self.sql_prompt_template,
        max_retries=self.sql_max_retries,
      )
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
    system_prompt: str | None = None,
  ) -> str:
    system = system_prompt or self.system_prompt
    context = "\n\n---\n\n".join(render_block(r.block) for r in retrieved)

    if self.prompt_builder is not None:
      return self.prompt_builder(
        system, context, query, sql=sql, sql_result=sql_result
      )

    sql_section = ""
    if sql and sql_result is not None:
      sql_section = (
        "\n\nComputed result (SQL was executed against the full parsed "
        f"tables, so it is exact):\nSQL: {sql}\nResult: {sql_result}\n"
        "Trust this computed result over manual arithmetic."
      )
    return (
      f"{system}\n\n{FORMAT_CONTRACT}\n\nContext:\n{context}{sql_section}\n\n"
      f"Question: {query}\n\nAnswer:"
    )

  def query(
    self,
    question: str,
    top_k: int = 3,
    system_prompt: str | None = None,
  ) -> QueryResult:
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

    prompt = self.build_prompt(
      question, retrieved, sql, sql_result, system_prompt=system_prompt
    )
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
