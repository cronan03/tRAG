"""Compute: ephemeral DuckDB sandbox + text-to-SQL agent."""

from tablerag.compute.sandbox import TableSandbox
from tablerag.compute.sql_agent import SQLAgent

__all__ = ["TableSandbox", "SQLAgent"]
