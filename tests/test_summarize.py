"""Pluggable summarizers for dual-vector indexing."""

from __future__ import annotations

from tablerag import TableRAGPipeline
from tablerag.generate import CallableGenerator
from tablerag.index.dual_vector import DualVectorIndex
from tablerag.index.embedder import HashEmbedder
from tablerag.models import TableBlock, TextBlock
from tablerag.summarize import (
  DEFAULT_MAX_DISTINCT_VALUES,
  DEFAULT_MAX_TEXT_CHARS,
  DeterministicSummarizer,
  LLMSummarizer,
  summarize_block,
)

TABLE = TableBlock(
  markdown="| region | net_rev_usd |\n| --- | --- |\n| NA-EAST | 100 |\n| NA-WEST | 200 |",
  headers=["region", "net_rev_usd"],
  rows=[["NA-EAST", "100"], ["NA-WEST", "200"]],
  section="SALES",
  source_format="pipe",
)


def test_summarize_block_default_matches_deterministic():
  summary = summarize_block(TABLE)
  assert "Table from section: SALES" in summary
  assert "region values: NA-EAST, NA-WEST" in summary
  assert summary == DeterministicSummarizer()(TABLE)


def test_deterministic_summarizer_respects_max_distinct_values():
  wide = TableBlock(
    markdown="",
    headers=["sku"],
    rows=[[f"SKU-{i}"] for i in range(50)],
    section="INV",
    source_format="pipe",
  )
  default = DeterministicSummarizer()(wide)
  assert default.count("SKU-") == DEFAULT_MAX_DISTINCT_VALUES

  tight = DeterministicSummarizer(max_distinct_values=5)(wide)
  assert tight.count("SKU-") == 5


def test_deterministic_summarizer_respects_max_text_chars():
  text = TextBlock(content="x" * 5000, section="NOTES")
  short = DeterministicSummarizer(max_text_chars=100)(text)
  assert len(short) < 200  # section prefix + 100 chars
  assert "x" * 100 in short
  assert "x" * 101 not in short
  assert DEFAULT_MAX_TEXT_CHARS == 1200


def test_custom_summarizer_callable_used_by_index():
  seen = []

  def my_summarizer(block):
    seen.append(block.doc_id)
    return f"CUSTOM::{block.section}"

  index = DualVectorIndex(HashEmbedder(), summarizer=my_summarizer)
  index.add_blocks([TABLE])
  assert seen == [TABLE.doc_id]
  assert index.summary_for(TABLE.doc_id) == "CUSTOM::SALES"


def test_pipeline_forwards_custom_summarizer():
  def my_summarizer(block):
    return f"PIPE::{block.kind}::{getattr(block, 'section', '')}"

  pipe = TableRAGPipeline(
    embedder=HashEmbedder(),
    enable_compute=False,
    summarizer=my_summarizer,
  )
  doc = """=== SALES ===
| region | rev |
| --- | --- |
| NA | 1 |
"""
  chunks = pipe.ingest(doc)
  assert chunks
  assert pipe.index.summary_for(chunks[0].doc_id).startswith("PIPE::")


def test_llm_summarizer_calls_generator():
  prompts = []

  def fake_llm(prompt: str) -> str:
    prompts.append(prompt)
    return "Narrative: weekly regional revenue table for NA stores."

  summarizer = LLMSummarizer(CallableGenerator(fake_llm, model="test"))
  summary = summarizer(TABLE)
  assert summary.startswith("Narrative:")
  assert len(prompts) == 1
  assert "SALES" in prompts[0]
  assert "net_rev_usd" in prompts[0] or "NA-EAST" in prompts[0]


def test_llm_summarizer_custom_prompt_template():
  prompts = []

  def fake_llm(prompt: str) -> str:
    prompts.append(prompt)
    return "ok"

  summarizer = LLMSummarizer(
    CallableGenerator(fake_llm),
    prompt_template="SUMMARIZE:\n{block}\nEND",
  )
  summarizer(TABLE)
  assert prompts[0].startswith("SUMMARIZE:")
  assert prompts[0].endswith("END")
