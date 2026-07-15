# tablerag Configurability & Robustness Issues

Tracker for hardcoded behavior and robustness gaps found in the codebase audit.
The theme: tunable "policy" decisions were baked in as module constants while
only structural parameters got exposed. Each item lists location, problem,
proposed fix, and status. Fix top-to-bottom.

Legend: [ ] todo  [~] in progress  [x] done

Detailed design for the prompt items (T1-1, T1-4) lives in
[docs/PROMPT_CUSTOMIZATION_PLAN.md](docs/PROMPT_CUSTOMIZATION_PLAN.md).

---

## Tier 1 - User-facing behavior the user cannot change

### [x] T1-1: SQL agent prompt + retry count hardcoded
- Location: [tablerag/compute/sql_agent.py](tablerag/compute/sql_agent.py) (`SQL_PROMPT`, `RETRY_SUFFIX`, retry loop in `answer()`).
- Problem: no way to inject domain SQL rules (fiscal calendars, unit conventions, business definitions); retry count frozen at 1.
- Fix (shipped): `sql_instructions` (appended; base safety contract intact), `sql_examples` (few-shot pairs), `sql_prompt_template` (full override with `{schema}`/`{question}`/`{instructions}` slots), `sql_max_retries` (default 1, 0 disables). Wired through the pipeline's `sql_agent` property. Tests in `tests/test_prompts.py`; docs in SDK.md "Customizing the SQL agent".

### [ ] T1-2: Query router is English-only regex with no override
- Location: [tablerag/route/classifier.py](tablerag/route/classifier.py) lines 12-43 (`_AGGREGATION_PATTERNS`); `pipeline.query()` has no manual route override.
- Problem: non-English queries always route to `lookup`, silently losing the compute/SQL feature; users cannot add domain trigger words or remove noisy ones; no escape hatch when the classifier is wrong.
- Fix:
  - Add `route=` override to `query()` (`"compute"` / `"lookup"` / `None`=auto).
  - Add `extra_compute_patterns` / `disable_patterns`, or accept a custom classifier callable, on `TableRAGPipeline`.
  - Document a way to bypass regex entirely (e.g. `classifier=my_fn`).

### [ ] T1-3: Summarization is deterministic-only and non-pluggable
- Location: [tablerag/summarize.py](tablerag/summarize.py); called directly at [tablerag/index/dual_vector.py](tablerag/index/dual_vector.py) line 64. Constants `MAX_DISTINCT_VALUES = 24`, `MAX_TEXT_CHARS = 1200`.
- Problem: no seam to enable LLM table summarization (the "dual-vector" industry-standard pattern the product is based on); summary caps frozen and they shape retrieval quality.
- Fix: a `summarizer=` callable slot on `DualVectorIndex` / `TableRAGPipeline` (default = current deterministic `summarize_block`); expose the two caps as params. Optionally ship an `LLMSummarizer` using the pipeline's generator.

### [x] T1-4: Generation system prompt + prompt assembly hardcoded
- Location: [tablerag/pipeline.py](tablerag/pipeline.py) (`SYSTEM_PROMPT`, `build_prompt()`, `render_block()`).
- Problem: no persona/tone/guardrails control, English-only, fixed layout and block rendering; the original complaint that kicked off this audit.
- Fix (shipped): `system_prompt` (constructor + per-query `query(..., system_prompt=...)`), `prompt_builder` callable for full layout control, persona split from the retained tablerag-owned `FORMAT_CONTRACT`. Tests in `tests/test_prompts.py`; docs in SDK.md "Customizing prompts".
- Deferred: injectable `render_block` (custom per-block rendering, e.g. HTML tables) — `prompt_builder` receives the context already joined.

---

## Tier 2 - Robustness gaps that will bite real users

### [ ] T2-1: Lexical scorer is ASCII-only
- Location: [tablerag/index/lexical.py](tablerag/index/lexical.py) line 13 (`_TOKEN_RE = [a-z0-9_\-\.]+`); frozen prefix weight `0.8`, min length `3`.
- Problem: non-Latin/accented text tokenizes to nothing, so the lexical half of hybrid scoring contributes ~0 and default `lexical_weight=0.5` effectively halves the semantic score for those corpora.
- Fix: Unicode-aware tokenizer (`\w` with `re.UNICODE`, or casefold); expose prefix weight and min length as config.

### [ ] T2-2: Number/date parsing in the sandbox is strict-format
- Location: [tablerag/compute/sandbox.py](tablerag/compute/sandbox.py) lines 18-20 (`_INT_RE`, `_FLOAT_RE`, `_DATE_RE`).
- Problem: does not recognize `391,005.15`, `$1,200`, `(4.5)` accounting negatives, `45%`, or non-ISO dates; such columns type as VARCHAR and `SUM()`/`AVG()` then fail, silently degrading compute to lookup.
- Fix: lenient coercion (strip thousands separators / currency / percent, handle parens-negatives) during type inference and load; add `TRY_CAST` guidance to the SQL prompt; consider a configurable coercion hook.

### [ ] T2-3: Section detection only knows two delimiter styles
- Location: [tablerag/parse/sections.py](tablerag/parse/sections.py) (`====` banners, `\n---\n` rules) - the formats of the repo's own test docs.
- Problem: real markdown uses `#`/`##` headings, which fall through to "whole document = one untitled section", degrading section titles, SQL table naming, and summaries at once.
- Fix: add a markdown-heading sectioner to the fallback chain in `split_sections()`; allow a custom sectioner to be injected.

### [ ] T2-4: GeminiEmbedder sends the whole corpus in one call
- Location: [tablerag/index/embedder.py](tablerag/index/embedder.py) line 46.
- Problem: no batching; a large document set hits the API per-request limit and fails outright.
- Fix: batch `embed()` into chunks (configurable `batch_size`), concatenate results.

---

## Tier 3 - Frozen tuning constants (promote to keyword args)

### [ ] T3-1: Retrieval candidate pool
- Location: [tablerag/index/dual_vector.py](tablerag/index/dual_vector.py) lines 29-30 (`CANDIDATE_MULTIPLIER = 4`, `MIN_CANDIDATES = 50`).
- Fix: constructor params; affects retrieval quality vs cost on large corpora.

### [ ] T3-2: Schema sample-row count
- Location: [tablerag/compute/sandbox.py](tablerag/compute/sandbox.py) line 122 (`LIMIT 2`).
- Fix: configurable; more samples help the SQL LLM on messy data.

### [ ] T3-3: Identifier truncation length
- Location: [tablerag/compute/sandbox.py](tablerag/compute/sandbox.py) line 33 (`[:60]`).
- Fix: configurable; long section titles collide after truncation.

### [ ] T3-4: Minimum columns to count as a table
- Location: [tablerag/parse/tables.py](tablerag/parse/tables.py) line 19 (`MIN_COLUMNS = 2`).
- Fix: configurable; single-column lists can never be tables today.

### [ ] T3-5: SQL keyword blocklist
- Location: [tablerag/compute/sandbox.py](tablerag/compute/sandbox.py) lines 23-26 (`FORBIDDEN_SQL`).
- Fix: allow tightening/loosening; note `set` currently blocks legitimate strings.

---

## Suggested fix order

1. T1-4 + T1-1 together (prompt customization; see prompt plan).
2. T1-2 (router escape hatch - cheap, high value).
3. T1-3 (pluggable summarizer - unlocks the dual-vector LLM pattern).
4. T2-1 .. T2-4 (robustness pass).
5. T3-x opportunistically as those files are touched.

## Non-issues (already fine)

- Generator/embedder provider slots (model-agnostic; done).
- Chunking params, similarity, lexical_weight (already constructor args).
- `HashEmbedder(dim=)`, evals `rel_tol` (already configurable/exposed).
- Script-local constants like `API_DELAY_SEC` in benchmarks (not SDK).
