"""Text-to-SQL agent over the DuckDB sandbox.

One LLM call to write the SQL, deterministic execution, one retry on SQL
error with the error message fed back. No open-ended agentic loops.
"""

from __future__ import annotations

import logging
import re

from tablerag.compute.sandbox import TableSandbox
from tablerag.generate import GeminiGenerator

logger = logging.getLogger("tablerag")

SQL_PROMPT = """You are a SQL analyst. Write ONE DuckDB SELECT statement that answers the question using the tables below.

{schema}

Rules:
- Output ONLY the SQL statement, no explanation, no markdown fences.
- Use only tables and columns listed above (names are case-sensitive).
- If the question cannot be answered with these tables/columns, output exactly: NO_SQL
- Empty cells are NULL; aggregate functions skip NULLs automatically.

Question: {question}

SQL:"""

RETRY_SUFFIX = """

Your previous attempt failed:
SQL: {sql}
Error: {error}

Write a corrected DuckDB SELECT statement (or NO_SQL if impossible):"""

_FENCE_RE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)


class SQLAgent:
  def __init__(self, sandbox: TableSandbox, generator: GeminiGenerator) -> None:
    self.sandbox = sandbox
    self.generator = generator

  def _clean_sql(self, raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()

  def answer(self, question: str) -> tuple[str | None, list[dict] | None]:
    """Return (sql, rows) or (None, None) if SQL is not applicable/failed."""
    if len(self.sandbox) == 0:
      return None, None

    schema = self.sandbox.schema_description()
    prompt = SQL_PROMPT.format(schema=schema, question=question)

    sql = self._clean_sql(self.generator.generate(prompt))
    if not sql or sql.upper().startswith("NO_SQL"):
      logger.info("SQL agent: declined (NO_SQL)")
      return None, None

    try:
      return sql, self.sandbox.execute(sql)
    except Exception as exc:
      logger.warning("SQL failed (%s); retrying once", exc)
      retry_prompt = prompt + RETRY_SUFFIX.format(sql=sql, error=exc)
      sql = self._clean_sql(self.generator.generate(retry_prompt))
      if not sql or sql.upper().startswith("NO_SQL"):
        return None, None
      try:
        return sql, self.sandbox.execute(sql)
      except Exception as exc2:
        logger.warning("SQL retry also failed (%s); falling back to lookup", exc2)
        return None, None
