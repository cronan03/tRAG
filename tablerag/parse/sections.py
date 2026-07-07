"""Split raw document text into titled sections.

Supported section delimiters, tried in order:
1. `====` banner blocks (document2 style):
       ================================
       SECTION TITLE
       ================================
       body...
2. `\n---\n` horizontal rules (tables_context style); title = first line.
3. Fallback: the whole document as one untitled section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BANNER_RE = re.compile(r"^={4,}\s*$")


@dataclass
class Section:
  title: str
  body: str


def _split_banner_sections(text: str) -> list[Section] | None:
  lines = text.splitlines()
  banner_indices = [i for i, line in enumerate(lines) if _BANNER_RE.match(line)]
  if len(banner_indices) < 2:
    return None

  sections: list[Section] = []

  # Preamble before the first banner.
  preamble = "\n".join(lines[: banner_indices[0]]).strip()
  if preamble:
    sections.append(Section(title="(preamble)", body=preamble))

  i = 0
  while i < len(banner_indices) - 1:
    start, end = banner_indices[i], banner_indices[i + 1]
    between = lines[start + 1 : end]
    # A title banner pair encloses exactly one non-empty line.
    non_empty = [ln for ln in between if ln.strip()]
    if len(non_empty) == 1 and end - start <= 2:
      title = non_empty[0].strip()
      body_start = end + 1
      body_end = banner_indices[i + 2] if i + 2 < len(banner_indices) else len(lines)
      body = "\n".join(lines[body_start:body_end]).strip()
      if body or title:
        sections.append(Section(title=title, body=body))
      i += 2
    else:
      # Banner pair encloses content (not a title); treat as body of previous.
      body = "\n".join(between).strip()
      if body:
        sections.append(Section(title="(untitled)", body=body))
      i += 1

  return sections if sections else None


def _split_hr_sections(text: str) -> list[Section] | None:
  parts = [part.strip() for part in text.split("\n---\n") if part.strip()]
  if len(parts) < 2:
    return None
  sections = []
  for part in parts:
    first_line = part.splitlines()[0].strip()
    title = first_line.lstrip("# ").strip() if first_line.startswith("#") else first_line
    sections.append(Section(title=title, body=part))
  return sections


def split_sections(text: str) -> list[Section]:
  return (
    _split_banner_sections(text)
    or _split_hr_sections(text)
    or [Section(title="(document)", body=text.strip())]
  )
