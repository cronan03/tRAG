# Tier 1 Fix Plan: User-Configurable Prompts

Status: IMPLEMENTED (Part A: system prompt; Part B: SQL instructions)
Scope: make the generation system prompt and the SQL-agent instructions
user-configurable, instead of hardcoded module constants.

## Problem

tablerag is a product for end users, but two prompts that directly control
answer behavior are baked in as module constants with no override seam:

- Generation persona/instructions:
  ```21:29:tablerag/pipeline.py
  SYSTEM_PROMPT = (
    "You are a data analyst answering questions about business documents that "
    ...
  )
  ```
- The prompt layout that consumes it:
  ```256:259:tablerag/pipeline.py
      return (
        f"{SYSTEM_PROMPT}\n\nContext:\n{context}{sql_section}\n\n"
        f"Question: {query}\n\nAnswer:"
      )
  ```
- The text-to-SQL prompt + retry text, plus a frozen retry count of 1:
  ```17:37:tablerag/compute/sql_agent.py
  SQL_PROMPT = """You are a SQL analyst. Write ONE DuckDB SELECT statement ...
  Rules:
  - Output ONLY the SQL statement, no explanation, no markdown fences.
  - Use only tables and columns listed above (names are case-sensitive).
  - If the question cannot be answered ... output exactly: NO_SQL
  - Empty cells are NULL; aggregate functions skip NULLs automatically.
  ...
  """
  ```

Consequences: no domain persona/tone/guardrails, English-only wording, no way
to inject domain SQL semantics (fiscal calendars, unit conventions, business
definitions), and no control over layout or retry count.

## How LangChain does it (reference)

In the repo's own LangChain baseline, the prompt is user code with named
slots, not framework-hidden:

```32:35:rag_lc.py
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
  [
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
```

The developer always owns the wording; the framework supplies the template  
primitive. tablerag will adopt the same idea while keeping a sensible default  
so the 3-line quickstart still works: expose the prompt as an overridable  
input rather than a hidden constant. 

## Design principle: separate persona from structural contract

The current prompt conflates two concerns. Keep them separate:

- Persona / domain (tone, guardrails, "you are a data analyst") -> USER-owned.
- Structural contract that retrieval/compute depend on ("tables are markdown
with section titles + notes"; the SQL-trust note appended in `build_prompt`
at `pipeline.py:250-255`) -> tablerag-owned, retained internally so a user's
custom persona cannot accidentally break table reading.

A full-control escape hatch (`prompt_builder`) is provided for users who
explicitly want to own the entire layout.

## Part A: System prompt (generation) — IMPLEMENTED

Three levels of control (progressive disclosure). Default behavior unchanged.
Shipped: `DEFAULT_SYSTEM_PROMPT` + `SYSTEM_PROMPT` alias + `FORMAT_CONTRACT`
split in `tablerag/pipeline.py`; `system_prompt=` / `prompt_builder=`
constructor params; per-query `query(..., system_prompt=...)`; tests in
`tests/test_prompts.py`; docs in SDK.md ("Customizing prompts") and DESIGN.md.

### Level 1: `system_prompt` string (covers ~90%)

```python
pipeline = TableRAGPipeline(
    generator=...,
    system_prompt="You are a clinical data assistant. Cite the source row for "
                  "every value. Never infer a diagnosis.",
)
```

Implementation:

- Rename `SYSTEM_PROMPT` -> `DEFAULT_SYSTEM_PROMPT` in
[tablerag/pipeline.py](tablerag/pipeline.py); keep `SYSTEM_PROMPT` as a
backward-compatible alias.
- `__init__(..., system_prompt: str | None = None)`; store
`self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT`.
- `build_prompt()` reads `self.system_prompt` instead of the constant.

### Level 2: `prompt_builder` callable (full layout control)

For few-shot, non-English scaffolds, HTML table rendering, custom ordering.

```python
def my_prompt(system, context, question, sql=None, sql_result=None):
    return f"{system}\n\n<docs>\n{context}\n</docs>\nQ: {question}"

pipeline = TableRAGPipeline(generator=..., prompt_builder=my_prompt)
```

- Framework-neutral (works with any Generator, not just LangChain).
- A LangChain `ChatPromptTemplate` adapts in one line:
  `prompt_builder=lambda s, c, q, sql=None, sql_result=None: tmpl.format(system=s, context=c, question=q)`.
- When set, `build_prompt()` delegates to it; otherwise uses the default layout.

### Level 3: per-query override

```python
pipeline.query("...", system_prompt="Answer in French.")
```

- `query(question, top_k=3, system_prompt=None)`; falls back to
`self.system_prompt` when not provided. Useful for multi-tenant apps.

## Part B: SQL instructions (compute) — IMPLEMENTED

Shipped: `{instructions}`/`{examples}` slots in `SQL_PROMPT`;
`SQLAgent(instructions=, examples=, prompt_template=, max_retries=)`;
pipeline params `sql_instructions` / `sql_examples` / `sql_prompt_template` /
`sql_max_retries` forwarded via the `sql_agent` property; tests in
`tests/test_prompts.py`; docs in SDK.md ("Customizing the SQL agent") and
DESIGN.md.

Different default from the system prompt: append domain instructions rather
than full replacement, because the base Rules are a safety + parse contract
("output ONLY SQL", "SELECT only", "NO_SQL fallback") that `_clean_sql()` and
the read-only `sandbox.execute()` guardrails rely on. Most users need to ADD
domain semantics, not rewrite guardrails.

### Primary knob: `sql_instructions` (appended into the Rules section)

```python
pipeline = TableRAGPipeline(
    generator=...,
    sql_instructions=(
        "Revenue columns are in USD thousands; multiply by 1000 for absolute.\n"
        "Fiscal year starts in April (Q1 = Apr-Jun).\n"
        "Exclude rows where status = 'void'.\n"
        "`available` = on_hand - reserved; treat NULL reserved as 0."
    ),
)
```

These are the use-case-specific facts the raw schema cannot express: unit
conventions, fiscal calendars, business definitions, tenant filters.

### High-leverage add-on: `sql_examples` (few-shot)

```python
sql_examples=[("total net revenue for NA",
               "SELECT SUM(net_rev_usd) FROM north_america")]
```

Rendered into the prompt; biggest accuracy lever on quirky schemas.

### Power-user override: `sql_prompt_template`

Full replacement with `{schema}`, `{question}`, `{instructions}` slots.
Documented caveat: caller must preserve the "output only SQL / NO_SQL"
contract at their own risk.

### Expose frozen control-flow: `sql_max_retries`

Currently a hardcoded single retry in `SQLAgent.answer()`
([tablerag/compute/sql_agent.py](tablerag/compute/sql_agent.py) lines 63-75).
Make it `sql_max_retries: int = 1`.

### Injection path

`TableRAGPipeline.__init_`_ accepts `sql_instructions`, `sql_examples`,
`sql_prompt_template`, `sql_max_retries` and passes them when it lazily builds
the agent in the `sql_agent` property
([tablerag/pipeline.py](tablerag/pipeline.py) lines 146-152):

```python
SQLAgent(
    self.sandbox,
    self.generator,
    instructions=self.sql_instructions,
    examples=self.sql_examples,
    prompt_template=self.sql_prompt_template,
    max_retries=self.sql_max_retries,
)
```

`SQL_PROMPT` gains an `{instructions}` slot (empty by default so current
behavior is byte-for-byte unchanged when nothing is passed).

## Decisions

- System prompt default: KEEP the current tablerag default persona (backward
compatible). Users override via `system_prompt`.
- SQL customization shape: APPEND-by-default (`sql_instructions`) plus an
optional full override (`sql_prompt_template`). Preserves the safety
contract for the common case.

(Both are reversible; flip here if product direction changes.)

## Implementation checklist

1. [tablerag/pipeline.py](tablerag/pipeline.py)
  - `DEFAULT_SYSTEM_PROMPT` (+ `SYSTEM_PROMPT` alias).
  - Constructor params: `system_prompt`, `prompt_builder`, `sql_instructions`,
  `sql_examples`, `sql_prompt_template`, `sql_max_retries`.
  - `build_prompt()` uses `self.system_prompt` / delegates to `prompt_builder`;
  route the SQL-trust note through the same path.
  - `query()` gains optional per-call `system_prompt`.
  - `sql_agent` property forwards the SQL params.
2. [tablerag/compute/sql_agent.py](tablerag/compute/sql_agent.py)
  - Add `{instructions}` slot to `SQL_PROMPT`; render `examples` when present.
  - `SQLAgent.__init__(..., instructions=None, examples=None, prompt_template=None, max_retries=1)`; use `max_retries` in `answer()`.
3. Pass-throughs
  - `from_env(**kwargs)` already forwards.
  - Optional CLI flags: `--system-prompt-file`, `--sql-instructions-file` in
  [tablerag/cli.py](tablerag/cli.py).
4. Tests ([tests/](tests/), new `test_prompts.py`)
  - Use a `CallableGenerator` that captures the prompt string to assert:
    - default prompt unchanged (backward compat);
    - `system_prompt` reaches the LLM;
    - `prompt_builder` is used when set;
    - `sql_instructions` appears in the SQL prompt on a compute query;
    - full `sql_prompt_template` override works;
    - `sql_max_retries` is honored.
5. Docs
  - New "Customizing prompts" section in [docs/SDK.md](docs/SDK.md) with the
   LangChain side-by-side.
  - Update generation/compute sections in [docs/DESIGN.md](docs/DESIGN.md).

## Backward compatibility

All new parameters are optional with defaults that reproduce current output
exactly. `SYSTEM_PROMPT` remains importable via alias. No breaking changes.

## Out of scope (later tiers)

- Tier 1 remaining: query-router escape hatch / custom patterns; pluggable
summarizer (LLM-summary mode).
- Tier 2: Unicode-aware lexical tokenizer; lenient numeric/date coercion in the
sandbox; markdown-heading sectioner; embedder batching.
- Tier 3: promote frozen tuning constants (candidate multiplier, schema sample
rows, identifier truncation, min columns, SQL blocklist) to keyword args.

