"""Word-budget prose splitting with optional overlap.

TextBlocks within the budget pass through untouched. Oversized blocks are
split at sentence boundaries; a single sentence longer than the budget is
hard-split by words. Every slice keeps the section title (annotated with
its part number) so no fragment floats free of its context.
"""

from __future__ import annotations

import re

from tablerag.models import TextBlock

DEFAULT_MAX_WORDS = 300

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
  parts = _SENTENCE_SPLIT_RE.split(text)
  return [p for p in parts if p.strip()]


def _hard_split(words: list[str], max_words: int) -> list[list[str]]:
  return [words[i : i + max_words] for i in range(0, len(words), max_words)]


def split_text_block(
  block: TextBlock,
  *,
  max_words: int = DEFAULT_MAX_WORDS,
  overlap_words: int = 0,
) -> list[TextBlock]:
  """Split a TextBlock into slices of at most max_words words.

  Args:
    block: the prose block to split.
    max_words: word budget per slice (default 300).
    overlap_words: words carried over from the end of the previous slice
      into the start of the next (default 0). Must be < max_words.
  """
  if overlap_words >= max_words:
    raise ValueError("overlap_words must be smaller than max_words")

  all_words = block.content.split()
  if len(all_words) <= max_words:
    return [block]

  # Build word groups: sentences, with oversized sentences hard-split.
  groups: list[list[str]] = []
  for sentence in _sentences(block.content):
    words = sentence.split()
    if len(words) > max_words:
      groups.extend(_hard_split(words, max_words))
    else:
      groups.append(words)

  slices: list[list[str]] = []
  current: list[str] = []
  for group in groups:
    if current and len(current) + len(group) > max_words:
      slices.append(current)
      current = current[-overlap_words:] if overlap_words else []
    current.extend(group)
  if current:
    slices.append(current)

  return [
    TextBlock(
      content=" ".join(words),
      section=f"{block.section} (part {i + 1}/{len(slices)})",
      source=block.source,
    )
    for i, words in enumerate(slices)
  ]
