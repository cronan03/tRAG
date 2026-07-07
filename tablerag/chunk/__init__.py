"""Chunking: header-aware table splitting + word-budget prose splitting."""

from tablerag.chunk.table_splitter import split_table_block
from tablerag.chunk.text_splitter import split_text_block

__all__ = ["split_table_block", "split_text_block"]
