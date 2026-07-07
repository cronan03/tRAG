"""tablerag — table-aware RAG infrastructure.

Parse heterogeneous documents (prose + tables), normalize tables to canonical
markdown, index them with the dual-vector pattern (summary embeddings pointing
to raw table payloads), and optionally route quantitative queries to a DuckDB
sandbox.
"""

from tablerag.models import QueryResult, TableBlock, TextBlock
from tablerag.pipeline import TableRAGPipeline

__all__ = [
  "TableBlock",
  "TextBlock",
  "QueryResult",
  "TableRAGPipeline",
]

__version__ = "0.1.0"
