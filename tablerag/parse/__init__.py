"""Parsing: heterogeneous text -> sections -> TextBlock / TableBlock."""

from tablerag.parse.detector import parse_document
from tablerag.parse.sections import Section, split_sections
from tablerag.parse.tables import try_parse_table

__all__ = ["parse_document", "Section", "split_sections", "try_parse_table"]
