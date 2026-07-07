# tablerag SDK Reference

Table-aware RAG infrastructure: parse heterogeneous documents (prose + tables),
normalize tables to canonical markdown, index them with the dual-vector pattern
(summary embeddings pointing to raw table payloads), and route quantitative
queries to an ephemeral DuckDB SQL sandbox.

- [Install](#install)
- [Quick start](#quick-start)
- [`TableRAGPipeline`](#tableragpipeline) — the main entry point
  - [Constructor](#constructor)
  - [`ingest()` / `parse()` / `chunk()` / `index_blocks()`](#ingestion-methods)
  - [`query()`](#query)
- [Configuration cookbook](#configuration-cookbook)
  - [Choosing the embedding model](#choosing-the-embedding-model)
  - [In-memory vs external vector DB](#in-memory-vs-external-vector-db)
  - [Similarity method](#similarity-method)
  - [Chunk sizing (words / rows / overlap)](#chunk-sizing)
  - [Separating chunking from embedding](#separating-chunking-from-embedding)
- [Evaluation harness](#evaluation-harness)
- [Data models](#data-models)
- [LangChain integration](#langchain-integration)
- [Lower-level building blocks](#lower-level-building-blocks)

---

## Install

```bash
pip install -e .            # core (google-genai, duckdb)
pip install -e ".[langchain]"   # + LangChain adapter
```

Set credentials in `.env`:

```
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.1-flash-lite        # generation model (optional)
GEMINI_EMBED_MODEL=gemini-embedding-001   # embedding model (optional)
```

---

## Quick start

```python
from tablerag import TableRAGPipeline

pipeline = TableRAGPipeline()
pipeline.ingest("data/document2.txt")

result = pipeline.query(
    "What is the total net revenue in USD across all NA stores for 2025-06-14?"
)
print(result.answer)   # -> "... 1,965,043.30 USD."
print(result.route)    # -> "hybrid" (SQL was used)
print(result.sql)      # -> "SELECT SUM(net_rev_usd) FROM north_america WHERE ..."
```

---

## `TableRAGPipeline`

`from tablerag import TableRAGPipeline`

The high-level object. Owns the vector index, the DuckDB sandbox, and the
Gemini generation client.

### Constructor

```python
TableRAGPipeline(
    embedder=None,
    model=None,
    max_table_rows=50,
    enable_compute=True,
    embed_model=None,
    similarity="cosine",
    lexical_weight=0.5,
    max_text_words=300,
    text_overlap_words=0,
    max_table_words=None,
    table_overlap_rows=0,
    vector_backend=None,
    vectorstore=None,
)
```

All parameters are **optional** — `TableRAGPipeline()` works out of the box.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `embedder` | `Embedder` | `GeminiEmbedder(embed_model)` | Custom embedding backend. Any object with `embed(list[str]) -> list[list[float]]`. |
| `model` | `str` | `GEMINI_MODEL` env or `gemini-3.1-flash-lite` | Gemini **generation** model. |
| `max_table_rows` | `int` | `50` | Hard row cap per table slice (header re-injected on each slice). |
| `enable_compute` | `bool` | `True` | Enable the DuckDB sandbox + SQL routing for aggregation queries. Set `False` for retrieval-only (no `duckdb` needed at query time). |
| `embed_model` | `str` | `GEMINI_EMBED_MODEL` env or `gemini-embedding-001` | Gemini **embedding** model. Ignored if `embedder` or an external backend is supplied. |
| `similarity` | `str` | `"cosine"` | Vector similarity for the in-memory backend: `"cosine"`, `"dot"`, or `"euclidean"`. |
| `lexical_weight` | `float` | `0.5` | Blend of exact-token (IDF) overlap vs semantic score at retrieval, `0.0`–`1.0`. `0` = pure semantic. |
| `max_text_words` | `int` | `300` | Word budget per prose chunk. |
| `text_overlap_words` | `int` | `0` | Words repeated between consecutive prose chunks. Must be `< max_text_words`. |
| `max_table_words` | `int` | `None` | Optional word budget per table slice; derives a row cap from average row width (stricter of this and `max_table_rows` wins). |
| `table_overlap_rows` | `int` | `0` | Rows repeated between consecutive table slices. Must be `< max_table_rows`. |
| `vector_backend` | `VectorBackend` | `None` | Custom vector backend (overrides the in-memory one). |
| `vectorstore` | LangChain `VectorStore` | `None` | Shorthand for `vector_backend=LangChainVectorStoreBackend(vectorstore)`. When set, embedding is done by the store, so `embed_model`/`similarity` are ignored. |

> Precedence for the vector side: `vectorstore` → `vector_backend` → in-memory
> backend built from `embedder`/`embed_model` + `similarity`.

### Ingestion methods

`ingest()` is the one-shot convenience. It is exactly
`parse()` → `chunk()` → `index_blocks()`, which you can also call
separately (see [Separating chunking from embedding](#separating-chunking-from-embedding)).

#### `ingest(path_or_text)`

Parse, chunk, and index in one call.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `path_or_text` | `str \| Path` | **yes** | A file path, or raw document text. |

Returns `list[Block]` (the chunked blocks that were indexed).

#### `parse(path_or_text)`

Stage 1. Parse into `TextBlock`/`TableBlock`. No chunking, no API calls.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `path_or_text` | `str \| Path` | **yes** | File path or raw text. |

Returns `list[Block]`.

#### `chunk(blocks)`

Stage 2. Apply word/row budgets. Deterministic, no API calls.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `blocks` | `list[Block]` | **yes** | Output of `parse()`. |

Returns `list[Block]` (chunked).

#### `index_blocks(blocks, *, sandbox_tables=None)`

Stage 3. Summarize + embed into the vector index (**the only ingest stage
that calls the embedding API**) and load tables into the DuckDB sandbox.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `blocks` | `list[Block]` | **yes** | Chunked blocks to index for retrieval. |
| `sandbox_tables` | `list[TableBlock]` | no (default: tables in `blocks`) | Tables to load into DuckDB. Pass the **original unchunked** tables so SQL always sees complete tables. |

Returns `None`.

### `query(question, top_k=3)`

Route, retrieve, optionally run SQL, and generate an answer.

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `question` | `str` | **yes** | — | The user question. |
| `top_k` | `int` | no | `3` | Number of blocks to retrieve into context. |

Returns a [`QueryResult`](#queryresult).

**Flow:** `classify_query` labels the question `lookup` or `compute`. Both routes
retrieve `top_k` blocks. `compute` additionally asks the SQL agent for one
DuckDB `SELECT`; on success the route becomes `hybrid` and the exact result is
injected into the prompt, otherwise it falls back to `lookup`.

---

## Configuration cookbook

### Choosing the embedding model

```python
# By model name (uses GeminiEmbedder under the hood)
TableRAGPipeline(embed_model="gemini-embedding-001")

# Or a fully custom embedder (must implement .embed(list[str]) -> list[list[float]])
from tablerag.index.embedder import Embedder

class MyEmbedder:
    def embed(self, texts): ...

TableRAGPipeline(embedder=MyEmbedder())
```

`HashEmbedder` (deterministic, offline, no API) is available for tests and
quota-free local runs:

```python
from tablerag.index.embedder import HashEmbedder
TableRAGPipeline(embedder=HashEmbedder())
```

### In-memory vs external vector DB

Default is in-memory (zero infrastructure). To use an external/persistent
store, pass any LangChain-compatible `VectorStore`:

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

store = Chroma(collection_name="tables", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
pipeline = TableRAGPipeline(vectorstore=store)   # store embeds + persists summaries
```

In this mode tablerag writes **summaries + `doc_id` pointers** to your store and
keeps the raw tables in its own docstore; search resolves pointers back to raw
tables (the multi-vector pattern). For a custom (non-LangChain) store, implement
the [`VectorBackend`](#lower-level-building-blocks) protocol and pass
`vector_backend=`.

### Similarity method

In-memory backend only (external stores use their own configured metric):

```python
TableRAGPipeline(similarity="cosine")     # default, magnitude-invariant
TableRAGPipeline(similarity="dot")        # inner product
TableRAGPipeline(similarity="euclidean")  # 1/(1+distance)
```

Tune the semantic/lexical blend independently:

```python
TableRAGPipeline(lexical_weight=0.0)   # pure embedding similarity
TableRAGPipeline(lexical_weight=1.0)   # pure exact-token (IDF) overlap
```

### Chunk sizing

```python
TableRAGPipeline(
    max_text_words=500,        # bigger prose chunks
    text_overlap_words=50,     # sliding-window overlap for prose
    max_table_rows=100,        # more rows per table slice
    max_table_words=1000,      # OR cap slices by word budget
    table_overlap_rows=5,      # repeat rows across table slices
)
```

Prose is split at sentence boundaries under a word budget; tables are split by
row groups with headers + section title + correction notes re-injected on every
slice. Overlap defaults to `0` for tables (rows are independent records).

### Separating chunking from embedding

Because `chunk()` is deterministic and free while `index_blocks()` makes the
embedding API calls, you can inspect or cache chunks before paying for
embeddings:

```python
pipeline = TableRAGPipeline(max_text_words=200)

blocks  = pipeline.parse("data/document2.txt")     # no API calls
chunked = pipeline.chunk(blocks)                    # no API calls
# ... inspect / filter / persist `chunked` here ...

originals = [b for b in blocks if b.kind == "table"]
pipeline.index_blocks(chunked, sandbox_tables=originals)   # embeds now
```

---

## Evaluation harness

`from tablerag.evals import Evaluator, load_doc2, load_wtq, load_t2ragbench`

Measures retrieval quality (**Recall@k**, **MRR@k**) and, optionally, answer
quality (**Exact Match** with numeric tolerance). Retrieval-only runs are free
(or fully offline with `HashEmbedder`); answer scoring makes one LLM call per
sample.

### Datasets

| Loader | Source | Offline | Notes |
| --- | --- | --- | --- |
| `load_doc2()` | built-in `data/document2.txt` | yes | 10 hand-verified stress queries. |
| `load_wtq(sample_size, split, config)` | `stanfordnlp/wikitablequestions` | no | Needs `pip install tablerag[evals]`. |
| `load_t2ragbench(sample_size, subset, split)` | `G4KMU/t2-ragbench` | no | `subset`: FinQA / ConvFinQA / TAT-DQA. |

Each returns `(samples, contexts)`: `samples: list[EvalSample]` and
`contexts: dict[context_id -> text]`. The `Evaluator` ingests each context under
`source=context_id`, and retrieval is scored a hit when a retrieved block
matches the sample's `golden_sections` (substring) or `golden_sources` (exact).

### `Evaluator(pipeline, top_k=3, generate=False, rel_tol=0.01, delay_sec=0.0)`

| Parameter | Default | Description |
| --- | --- | --- |
| `pipeline` | — | A configured `TableRAGPipeline` (its embedder/backend/chunking are what get evaluated). |
| `top_k` | `3` | Retrieval depth for Recall@k / MRR@k. |
| `generate` | `False` | Also run `query()` per sample and score answers (LLM calls). |
| `rel_tol` | `0.01` | Numeric tolerance for answer matching. |
| `delay_sec` | `0.0` | Sleep between generation calls (rate limits). |

`Evaluator.run(samples, contexts, dataset_name="") -> EvalReport`.

```python
from tablerag.evals import Evaluator, load_doc2
from tablerag.pipeline import TableRAGPipeline

samples, contexts = load_doc2()
report = Evaluator(TableRAGPipeline(), top_k=3, generate=True).run(samples, contexts)
print(report.recall_at_k, report.mrr_at_k, report.exact_match)
```

### CLI

```bash
python -m tablerag.evals doc2 --offline                    # free, retrieval-only
python -m tablerag.evals doc2 --generate --compute         # + answers + SQL route
python -m tablerag.evals wtq  --sample-size 50             # WikiTableQuestions
python -m tablerag.evals t2   --sample-size 25 --subset FinQA
```

Add `--out report.json` to save the full per-sample report. Latest built-in
`doc2` result (live Gemini embeddings + generation + compute): **Recall@3 90%,
MRR@3 0.78, Exact Match 100%**.

---

## Data models

`from tablerag.models import TextBlock, TableBlock, QueryResult, RetrievedBlock`

### `TextBlock`
| Field | Type | Description |
| --- | --- | --- |
| `content` | `str` | Prose text. |
| `section` | `str` | Section title it came from. |
| `doc_id` | `str` | UUID (auto). |
| `source` | `str` | Source filename. |
| `kind` | `str` | `"text"` (read-only). |

### `TableBlock`
| Field | Type | Description |
| --- | --- | --- |
| `markdown` | `str` | Canonical markdown table. |
| `headers` | `list[str]` | Column names. |
| `rows` | `list[list[str]]` | Row cells. |
| `section` | `str` | Section title. |
| `source_format` | `str` | `pipe` / `tsv` / `csv`. |
| `context_notes` | `str` | Nearby prose (e.g. correction footnotes). |
| `doc_id`, `source` | `str` | As above. |
| `kind`, `num_rows`, `num_cols` | — | Read-only properties. |

### `QueryResult`
| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | The question. |
| `answer` | `str` | Generated answer. |
| `route` | `str` | `lookup` / `compute` / `hybrid`. |
| `retrieved` | `list[RetrievedBlock]` | Retrieved blocks + scores. |
| `sql` | `str \| None` | Executed SQL (hybrid route). |
| `sql_result` | `list[dict] \| None` | SQL rows. |
| `prompt` | `str` | Final prompt sent to the LLM. |
| `model` | `str` | Generation model used. |

`RetrievedBlock` has `.block` (a `TextBlock`/`TableBlock`) and `.score` (float).

---

## LangChain integration

`from tablerag.integrations.langchain import TableRetrieverManager`

Wraps tablerag's ingest + retrieval as a native LangChain `BaseRetriever`,
replacing the manual `MultiVectorRetriever` + UUID + summary-chain boilerplate.

### `TableRetrieverManager(...)`

```python
TableRetrieverManager(
    vectorstore=None,
    embedder=None,
    max_table_rows=50,
    max_table_words=None,
    max_text_words=300,
    text_overlap_words=0,
    similarity="cosine",
    lexical_weight=0.5,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `vectorstore` | LangChain `VectorStore` | `None` | If set, summaries go here; raw tables stay in tablerag's docstore. If omitted, uses the internal in-memory index. |
| `embedder` | `Embedder \| Embeddings` | `GeminiEmbedder()` | Internal mode only. Accepts a tablerag `Embedder` **or** a LangChain `Embeddings`. |
| `max_table_rows`, `max_table_words`, `max_text_words`, `text_overlap_words` | | | Chunking, same semantics as the pipeline. |
| `similarity`, `lexical_weight` | | | Internal mode only (vectorstore mode ranks with the store's metric). |

**Methods**

| Method | Signature | Description |
| --- | --- | --- |
| `ingest` | `ingest(path_or_text) -> list[Block]` | Parse + chunk + index a file or text. |
| `ingest_tables` | `ingest_tables(list[str]) -> list[Block]` | Ingest raw table strings (markdown/TSV/CSV). |
| `search` | `search(query, k=3) -> list[Document]` | Retrieve raw blocks as LangChain `Document`s (metadata: `doc_id`, `kind`, `section`, `source`, `score`). |
| `as_retriever` | `as_retriever(k=3) -> BaseRetriever` | Native retriever for any LCEL chain. |

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

manager = TableRetrieverManager()
manager.ingest("data/document2.txt")
retriever = manager.as_retriever(k=3)

chain = (
    {"context": retriever | (lambda ds: "\n\n".join(d.page_content for d in ds)),
     "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Context:\n{context}\n\nQ: {question}")
    | ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    | StrOutputParser()
)
print(chain.invoke("corrected net_rev_eur for DE-442 on 2025-06-07?"))
```

See a full runnable example in [examples/langchain_table_rag.py](../examples/langchain_table_rag.py).

---

## Lower-level building blocks

For advanced/custom pipelines, the sub-packages are usable directly:

| Import | What it does |
| --- | --- |
| `from tablerag.parse import parse_document, split_sections, try_parse_table` | Text → blocks; section splitting; single-table detection. |
| `from tablerag.chunk import split_table_block, split_text_block` | Header-aware table splitting; word-budget prose splitting. |
| `from tablerag.summarize import summarize_block` | Deterministic searchable summary for a block. |
| `from tablerag.index import DualVectorIndex, DocStore, GeminiEmbedder, HashEmbedder` | Dual-vector index, docstore, embedders. |
| `from tablerag.index import InMemoryBackend, LangChainVectorStoreBackend, VectorBackend` | Vector backends + the protocol to implement your own. |
| `from tablerag.route import classify_query` | `lookup` vs `compute` classifier. |
| `from tablerag.compute import TableSandbox, SQLAgent` | Ephemeral DuckDB sandbox; text-to-SQL agent. |

### `VectorBackend` protocol

Implement these three methods to plug in any store:

```python
class VectorBackend(Protocol):
    def add(self, ids: list[str], summaries: list[str]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...  # (doc_id, score), higher=better
    def __len__(self) -> int: ...
```

### `DualVectorIndex(...)`

```python
DualVectorIndex(embedder=None, lexical_weight=0.5, backend=None, similarity="cosine")
```

| Parameter | Default | Description |
| --- | --- | --- |
| `embedder` | `None` | Required unless `backend` is given. |
| `lexical_weight` | `0.5` | Semantic/lexical blend. |
| `backend` | `None` | Custom `VectorBackend`; defaults to `InMemoryBackend(embedder, similarity)`. |
| `similarity` | `"cosine"` | `cosine` / `dot` / `euclidean` (in-memory backend). |

Methods: `add_blocks(blocks)`, `search(query, top_k=3) -> list[RetrievedBlock]`.
