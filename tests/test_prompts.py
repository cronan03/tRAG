"""User-configurable prompts (Part A: system prompt; Part B: SQL agent)."""

from __future__ import annotations

from tablerag import TableRAGPipeline
from tablerag.compute.sandbox import TableSandbox
from tablerag.compute.sql_agent import SQLAgent
from tablerag.generate import CallableGenerator
from tablerag.index.embedder import HashEmbedder
from tablerag.models import TableBlock
from tablerag.parse import parse_document
from tablerag.pipeline import DEFAULT_SYSTEM_PROMPT, FORMAT_CONTRACT, SYSTEM_PROMPT

DOC = """=== SALES ===
Weekly revenue by region.

| region | net_rev_usd |
| --- | --- |
| NA-EAST | 640220.10 |
| NA-WEST | 325044.80 |
"""


def make_pipeline(**kwargs) -> TableRAGPipeline:
  pipe = TableRAGPipeline(
    generator=CallableGenerator(lambda p: "ok", model="test-llm"),
    embedder=HashEmbedder(),
    enable_compute=False,
    **kwargs,
  )
  pipe.ingest(DOC)
  return pipe


def test_default_prompt_contains_persona_and_contract():
  pipe = make_pipeline()
  result = pipe.query("What is NA-WEST net revenue?")
  assert DEFAULT_SYSTEM_PROMPT in result.prompt
  assert FORMAT_CONTRACT in result.prompt
  assert "Question: What is NA-WEST net revenue?" in result.prompt
  # Backward-compat alias still points at the default persona.
  assert SYSTEM_PROMPT == DEFAULT_SYSTEM_PROMPT


def test_constructor_system_prompt_replaces_persona_keeps_contract():
  persona = "You are a clinical data assistant. Cite the source row."
  pipe = make_pipeline(system_prompt=persona)
  result = pipe.query("What is NA-WEST net revenue?")
  assert persona in result.prompt
  assert DEFAULT_SYSTEM_PROMPT not in result.prompt
  # The tablerag-owned format contract survives a custom persona.
  assert FORMAT_CONTRACT in result.prompt


def test_per_query_system_prompt_overrides_constructor():
  pipe = make_pipeline(system_prompt="Constructor persona.")
  result = pipe.query("What is NA-WEST net revenue?", system_prompt="Answer in French.")
  assert "Answer in French." in result.prompt
  assert "Constructor persona." not in result.prompt
  # Next query without an override falls back to the constructor persona.
  result2 = pipe.query("What is NA-EAST net revenue?")
  assert "Constructor persona." in result2.prompt


def test_prompt_builder_owns_entire_layout():
  def my_builder(system, context, question, sql=None, sql_result=None):
    return f"<sys>{system}</sys>\n<docs>{context}</docs>\nQ: {question}"

  pipe = make_pipeline(system_prompt="Persona.", prompt_builder=my_builder)
  result = pipe.query("What is NA-WEST net revenue?")
  assert result.prompt.startswith("<sys>Persona.</sys>")
  assert "<docs>" in result.prompt
  assert "325044.80" in result.prompt  # retrieved context flowed through
  # Default scaffold is fully replaced.
  assert FORMAT_CONTRACT not in result.prompt
  assert "Answer:" not in result.prompt


def test_prompt_builder_receives_sql_kwargs():
  captured = {}

  def my_builder(system, context, question, sql=None, sql_result=None):
    captured["sql"] = sql
    captured["sql_result"] = sql_result
    return "prompt"

  pipe = make_pipeline(prompt_builder=my_builder)
  pipe.build_prompt(
    "q", [], sql="SELECT 1", sql_result=[{"x": 1}]
  )
  assert captured["sql"] == "SELECT 1"
  assert captured["sql_result"] == [{"x": 1}]


# ------------------------------------------------------ Part B: SQL agent
class RecordingGenerator:
  """Returns canned responses in order; records every prompt it sees."""

  model = "test-sql-llm"

  def __init__(self, responses: list[str]):
    self._responses = responses
    self.prompts: list[str] = []

  def generate(self, prompt: str) -> str:
    self.prompts.append(prompt)
    index = min(len(self.prompts) - 1, len(self._responses) - 1)
    return self._responses[index]


def make_sandbox() -> TableSandbox:
  blocks = parse_document(DOC, source="t")
  tables = [b for b in blocks if isinstance(b, TableBlock)]
  sandbox = TableSandbox()
  sandbox.load_tables(tables)
  return sandbox


def test_sql_default_prompt_has_no_extra_sections():
  gen = RecordingGenerator(["SELECT 42 AS x"])
  agent = SQLAgent(make_sandbox(), gen)
  sql, rows = agent.answer("total?")
  assert sql == "SELECT 42 AS x"
  assert rows == [{"x": 42}]
  prompt = gen.prompts[0]
  assert "Output ONLY the SQL statement" in prompt  # base contract intact
  assert "Domain rules" not in prompt
  assert "Examples:" not in prompt


def test_sql_instructions_are_appended_not_replacing():
  gen = RecordingGenerator(["SELECT 42 AS x"])
  agent = SQLAgent(
    make_sandbox(),
    gen,
    instructions="Revenue columns are in USD thousands; multiply by 1000.",
  )
  agent.answer("total?")
  prompt = gen.prompts[0]
  assert "USD thousands" in prompt
  assert "Domain rules (use-case specific" in prompt
  # Safety contract survives.
  assert "Output ONLY the SQL statement" in prompt
  assert "output exactly: NO_SQL" in prompt


def test_sql_examples_rendered_as_few_shot():
  gen = RecordingGenerator(["SELECT 42 AS x"])
  agent = SQLAgent(
    make_sandbox(),
    gen,
    examples=[("total NA revenue", "SELECT SUM(net_rev_usd) FROM sales")],
  )
  agent.answer("total?")
  prompt = gen.prompts[0]
  assert "Examples:" in prompt
  assert "Q: total NA revenue" in prompt
  assert "SQL: SELECT SUM(net_rev_usd) FROM sales" in prompt


def test_sql_prompt_template_full_override():
  gen = RecordingGenerator(["SELECT 42 AS x"])
  template = "MY TEMPLATE\n{schema}\nRULES: {instructions}\nQ: {question}\nSQL:"
  agent = SQLAgent(
    make_sandbox(), gen, prompt_template=template, instructions="be careful"
  )
  agent.answer("total?")
  prompt = gen.prompts[0]
  assert prompt.startswith("MY TEMPLATE")
  assert "RULES: be careful" in prompt
  assert "Output ONLY the SQL statement" not in prompt  # default gone


def test_sql_max_retries_zero_gives_up_after_first_failure():
  gen = RecordingGenerator(["SELECT nope FROM missing_table"])
  agent = SQLAgent(make_sandbox(), gen, max_retries=0)
  assert agent.answer("total?") == (None, None)
  assert len(gen.prompts) == 1  # no retry generation


def test_sql_max_retries_two_allows_three_attempts():
  gen = RecordingGenerator(
    [
      "SELECT nope FROM missing_table",
      "SELECT still FROM broken",
      "SELECT 7 AS x",
    ]
  )
  agent = SQLAgent(make_sandbox(), gen, max_retries=2)
  sql, rows = agent.answer("total?")
  assert rows == [{"x": 7}]
  assert len(gen.prompts) == 3
  # Retry prompts carry the failure feedback.
  assert "Your previous attempt failed" in gen.prompts[1]


def test_pipeline_forwards_sql_options_end_to_end():
  prompts: list[str] = []

  def llm(prompt: str) -> str:
    prompts.append(prompt)
    if prompt.rstrip().endswith("SQL:"):
      return "SELECT 42 AS total"  # table-independent, always executes
    return "final answer"

  pipe = TableRAGPipeline(
    generator=CallableGenerator(llm, model="test-llm"),
    embedder=HashEmbedder(),
    sql_instructions="Fiscal year starts in April.",
    sql_max_retries=0,
  )
  pipe.ingest(DOC)
  result = pipe.query("What is the total net revenue across all regions?")

  assert result.route == "hybrid"
  sql_prompts = [p for p in prompts if p.rstrip().endswith("SQL:")]
  assert sql_prompts and "Fiscal year starts in April." in sql_prompts[0]
  assert result.sql_result == [{"total": 42}]
