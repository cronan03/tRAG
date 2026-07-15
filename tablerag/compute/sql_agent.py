"""Text-to-SQL agent over the DuckDB sandbox.

One LLM call to write the SQL, deterministic execution, configurable retries
on SQL error with the error message fed back (default 1). No open-ended
agentic loops.

The prompt is user-customizable in two ways:
- `instructions` / `examples`: APPENDED to the base prompt. The base Rules are
  a safety + parse contract ("output ONLY SQL", "SELECT only", "NO_SQL"
  fallback) that _clean_sql() and the read-only sandbox rely on, so the common
  case adds domain semantics without touching the guardrails.
- `prompt_template`: full replacement with {schema}, {question}, and
  {instructions} slots, for callers who explicitly want to own the contract.
"""

from __future__ import annotations

import logging
import re

from tablerag.compute.sandbox import TableSandbox
from tablerag.generate import Generator

logger = logging.getLogger("tablerag")

SQL_PROMPT = """You are a SQL analyst. Write ONE DuckDB SELECT statement that answers the question using the tables below.

{schema}

Rules:
- Output ONLY the SQL statement, no explanation, no markdown fences.
- Use only tables and columns listed above (names are case-sensitive).
- If the question cannot be answered with these tables/columns, output exactly: NO_SQL
- Empty cells are NULL; aggregate functions skip NULLs automatically.{instructions}{examples}

Question: {question}

SQL:"""

RETRY_SUFFIX = """

Your previous attempt failed:
SQL: {sql}
Error: {error}

Write a corrected DuckDB SELECT statement (or NO_SQL if impossible):"""

_FENCE_RE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)


class SQLAgent:
  def __init__(
    self,
    sandbox: TableSandbox,
    generator: Generator,
    instructions: str | None = None,
    examples: list[tuple[str, str]] | None = None,
    prompt_template: str | None = None,
    max_retries: int = 1,
  ) -> None:
    """
    Args:
      sandbox: the DuckDB sandbox to execute against.
      generator: any Generator (writes the SQL).
      instructions: use-case domain rules appended to the base prompt (unit
        conventions, fiscal calendars, business definitions, tenant filters).
      examples: few-shot (question, sql) pairs rendered into the prompt.
      prompt_template: full prompt replacement with {schema}, {question},
        {instructions} slots. Caller owns the "output only SQL / NO_SQL"
        contract when overriding.
      max_retries: how many corrected attempts after a failed execution
        (each feeds the error back to the LLM). 0 disables retrying.
    """
    self.sandbox = sandbox
    self.generator = generator
    self.instructions = instructions
    self.examples = examples
    self.prompt_template = prompt_template
    self.max_retries = max_retries

  def _build_prompt(self, schema: str, question: str) -> str:
    if self.prompt_template is not None:
      return self.prompt_template.format(
        schema=schema, question=question, instructions=self.instructions or ""
      )

    instructions_block = ""
    if self.instructions:
      instructions_block = (
        "\n\nDomain rules (use-case specific, follow strictly):\n"
        + self.instructions.strip()
      )
    examples_block = ""
    if self.examples:
      rendered = "\n\n".join(f"Q: {q}\nSQL: {s}" for q, s in self.examples)
      examples_block = "\n\nExamples:\n" + rendered
    return SQL_PROMPT.format(
      schema=schema,
      question=question,
      instructions=instructions_block,
      examples=examples_block,
    )

  def _clean_sql(self, raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()

  def answer(self, question: str) -> tuple[str | None, list[dict] | None]:
    """Return (sql, rows) or (None, None) if SQL is not applicable/failed."""
    if len(self.sandbox) == 0:
      return None, None

    schema = self.sandbox.schema_description()
    prompt = self._build_prompt(schema, question)

    sql = self._clean_sql(self.generator.generate(prompt))
    if not sql or sql.upper().startswith("NO_SQL"):
      logger.info("SQL agent: declined (NO_SQL)")
      return None, None

    attempts = 0
    while True:
      try:
        return sql, self.sandbox.execute(sql)
      except Exception as exc:
        attempts += 1
        if attempts > self.max_retries:
          logger.warning(
            "SQL failed (%s); retries exhausted (%d); falling back to lookup",
            exc,
            self.max_retries,
          )
          return None, None
        logger.warning(
          "SQL failed (%s); retry %d/%d", exc, attempts, self.max_retries
        )
        retry_prompt = prompt + RETRY_SUFFIX.format(sql=sql, error=exc)
        sql = self._clean_sql(self.generator.generate(retry_prompt))
        if not sql or sql.upper().startswith("NO_SQL"):
          return None, None
