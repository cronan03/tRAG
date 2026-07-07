"""Key-value store mapping doc_id -> raw block payload.

In-memory dict with optional JSON persistence. This is the "knowledge vault"
half of the dual-vector pattern: raw tables live here unfragmented, while the
vector index only holds their searchable summaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tablerag.models import Block, TableBlock, TextBlock


def _block_to_dict(block: Block) -> dict:
  data = asdict(block)
  data["kind"] = block.kind
  return data


def _block_from_dict(data: dict) -> Block:
  kind = data.pop("kind")
  if kind == "table":
    return TableBlock(**data)
  return TextBlock(**data)


class DocStore:
  def __init__(self) -> None:
    self._blocks: dict[str, Block] = {}

  def put(self, block: Block) -> None:
    self._blocks[block.doc_id] = block

  def get(self, doc_id: str) -> Block:
    return self._blocks[doc_id]

  def __len__(self) -> int:
    return len(self._blocks)

  def __contains__(self, doc_id: str) -> bool:
    return doc_id in self._blocks

  def all_blocks(self) -> list[Block]:
    return list(self._blocks.values())

  def save(self, path: Path) -> None:
    payload = [_block_to_dict(b) for b in self._blocks.values()]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  @classmethod
  def load(cls, path: Path) -> DocStore:
    store = cls()
    for data in json.loads(path.read_text(encoding="utf-8")):
      store.put(_block_from_dict(data))
    return store
